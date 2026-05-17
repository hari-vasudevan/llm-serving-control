#!/usr/bin/env python3
"""
Chapter 11 Modal vLLM token-budget measurement wrapper.

This file starts from the Chapter 10 wrapper. In Chapter 11 Phase 1 it drives
open-loop token-budget sweeps by writing fixed admission_fraction values to
the scheduler control file, then measuring TTFT, throughput, latency, GPU
power, and energy per request.

The implementation is intentionally verbose so that `tail` on the Modal logs
shows what the Mac sent, what the server received, queue evolution, dispatch
activity, latency samples, and native vLLM metrics.
"""

import argparse
import collections
import json
import math
import os
import random
import re
import statistics
import threading
import time
import urllib.parse
from dataclasses import dataclass, asdict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import requests

try:
    import pynvml
except Exception:  # pragma: no cover - depends on NVIDIA runtime
    pynvml = None


@dataclass
class QueueItem:
    request_id: str
    prompt: str
    prompt_chars: int
    prompt_repeat: int
    max_tokens: int
    temperature: float
    source: str
    client_ts: str
    enqueued_wall: str
    enqueued_perf: float


TRACE_PREFIX = "CH11"
LOCK = threading.Lock()
FIFO = collections.deque()
RECENT_EVENTS = collections.deque(maxlen=500)
RECENT_TICKS = collections.deque(maxlen=120)
L_MEAN_BUF = collections.deque(maxlen=300)
TTFT_BUF = collections.deque(maxlen=300)
QWAIT_BUF = collections.deque(maxlen=300)
ARRIVAL_TS = collections.deque(maxlen=400)
REQ_COUNTER = 0
TICK = 0
DISPATCHED = 0
COMPLETED = 0
ERRORS = 0
B = 4
DT = 1.0
B_MIN = 1
B_MAX = 50
MAX_TOKENS_DEFAULT = 32
PROMPT_REPEAT_DEFAULT = 192
TIMEOUT = 180.0
MODEL = "Qwen/Qwen2.5-3B-Instruct"
BACKEND_URL = "http://127.0.0.1:8001"
METRICS_URL = "http://127.0.0.1:8001/metrics"
HEALTH_URL = "http://127.0.0.1:8001/health"
API_KEY = ""
CONTROL_FILE = "/tmp/ch11_scheduler_control.json"
STATUS_FILE = "/tmp/ch11_scheduler_status.json"
LAST_CONTROL_SOURCE = "startup"
LAST_CONTROL_TS = ""
QUEUE_AREA = 0.0
QUEUE_LAST_TS = time.perf_counter()
TICK_ARRIVALS = 0
TICK_COMPLETIONS = 0
TICK_Q_MAX = 0
LAST_TICK_SUMMARY = {
    "tick": 0,
    "q_mean_tick": 0.0,
    "q_max_tick": 0,
    "arrivals_tick": 0,
    "completions_tick": 0,
    "service_rate_tick": 0.0,
    "lambda_tick": 0.0,
}
PROXY_LAT_BUF = collections.deque(maxlen=1000)
PROXY_TTFT_BUF = collections.deque(maxlen=1000)
PROXY_ERRORS = 0
NVML_HANDLE = None
CONTROL_WRITE_LOCK = threading.Lock()


def log(message):
    print(f"[{TRACE_PREFIX} {datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def short_prompt(text, limit=72):
    clean = text.replace("\n", " ").strip()
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def parse_metrics_text(raw_text):
    metrics = {}
    for line in raw_text.splitlines():
        if not line or line.startswith("#"):
            continue
        clean = re.sub(r"\{[^}]*\}", "", line).strip()
        parts = clean.split()
        if len(parts) < 2:
            continue
        try:
            metrics[parts[0]] = metrics.get(parts[0], 0.0) + float(parts[1])
        except ValueError:
            continue
    return metrics


def hist_mean_ms(metrics, stem):
    total = metrics.get(f"{stem}_sum")
    count = metrics.get(f"{stem}_count")
    if total is None or count in (None, 0):
        return None
    return round((total / count) * 1000.0, 2)


def fetch_backend_metrics():
    try:
        raw = requests.get(METRICS_URL, timeout=5).text
        return parse_metrics_text(raw)
    except Exception as exc:
        log(f"metrics fetch failed: {exc}")
        return {}


def gpu_snapshot():
    global NVML_HANDLE
    if pynvml is None:
        return {}
    try:
        if NVML_HANDLE is None:
            pynvml.nvmlInit()
            NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(NVML_HANDLE)
        mem = pynvml.nvmlDeviceGetMemoryInfo(NVML_HANDLE)
        power_w = pynvml.nvmlDeviceGetPowerUsage(NVML_HANDLE) / 1000.0
        out = {
            "gpu_power_w": round(power_w, 3),
            "gpu_util_percent": float(util.gpu),
            "gpu_mem_util_percent": float(util.memory),
            "gpu_memory_used_mb": round(mem.used / (1024.0 * 1024.0), 3),
        }
        try:
            out["gpu_temperature_c"] = float(pynvml.nvmlDeviceGetTemperature(NVML_HANDLE, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
        return out
    except Exception as exc:
        return {"gpu_power_error": str(exc)}


def scheduler_status():
    try:
        with open(STATUS_FILE) as f:
            payload = json.load(f)
        return {
            "scheduler_mode": payload.get("mode"),
            "scheduler_admission_fraction": payload.get("admission_fraction"),
            "scheduler_token_cap": payload.get("token_cap"),
            "scheduler_running_cap": payload.get("running_cap"),
            "scheduler_target_ttft_ms": payload.get("target_ttft_ms"),
            "scheduler_measured_ttft_ms": payload.get("measured_ttft_ms"),
            "scheduler_xi": payload.get("xi"),
        }
    except Exception:
        return {}


def write_scheduler_control(payload):
    with CONTROL_WRITE_LOCK:
        tmp = CONTROL_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, CONTROL_FILE)


def metric_delta_mean_ms(before, after, stem):
    d_sum = after.get(f"{stem}_sum", 0.0) - before.get(f"{stem}_sum", 0.0)
    d_count = after.get(f"{stem}_count", 0.0) - before.get(f"{stem}_count", 0.0)
    if d_count <= 0:
        return None
    return 1000.0 * d_sum / d_count


def percentile(values, pct):
    vals = sorted(v for v in values if v is not None and math.isfinite(v))
    if not vals:
        return None
    idx = min(len(vals) - 1, max(0, round((pct / 100.0) * (len(vals) - 1))))
    return vals[idx]


def integrate_power(samples):
    if len(samples) < 2:
        return None
    total = 0.0
    for a, b in zip(samples, samples[1:]):
        p0 = a.get("gpu_power_w")
        p1 = b.get("gpu_power_w")
        if p0 is None or p1 is None:
            continue
        total += 0.5 * (float(p0) + float(p1)) * max(0.0, float(b["t"]) - float(a["t"]))
    return total


QUESTION_BANK = [
    # Physics
    "What is the speed of light in a vacuum?",
    "How does a transistor work?",
    "What is quantum entanglement?",
    "What is the Doppler effect?",
    "What is entropy in thermodynamics?",
    "What is dark matter?",
    "What is the Higgs boson?",
    "What is a superconductor?",
    "What is the difference between AC and DC current?",
    "What is the Turing test?",
    "What is general relativity in simple terms?",
    "What is the difference between speed and velocity?",
    "What is quantum computing?",
    "What is the difference between fission and fusion?",
    "What is the multiverse hypothesis?",
    "How does a laser produce light?",
    "How does a gyroscope maintain orientation?",
    "How do fuel cells generate electricity?",
    "How does a heat pump work?",
    "How does a nuclear reactor generate electricity?",
    "How does nuclear fusion differ from fission in energy output?",
    "What is special relativity and how does it affect time?",
    "How does a particle accelerator work?",
    "What is antimatter and can it be stored?",
    "How does a scanning tunnelling microscope image individual atoms?",
    "What is wave-particle duality?",
    "How does the photoelectric effect work?",
    "What is Heisenberg's uncertainty principle?",
    "How does a plasma stay contained in a tokamak?",
    "What is Bose-Einstein condensate?",
    "How does acoustic levitation work?",
    "What is piezoelectricity?",
    "How does a thermoelectric generator convert heat to electricity?",
    "What is the Casimir effect?",
    "How does a Hall effect sensor work?",
    "What is magnetic reconnection?",
    "How does inertial confinement fusion work?",
    "What is a phonon?",
    "How does supercooling work?",
    "What is the difference between ferromagnetism and paramagnetism?",
    "How does a Stirling engine work?",
    "What is impedance in electrical circuits?",
    "How does a transformer change voltage?",
    "What is the skin effect in conductors?",
    "How does a capacitor store charge?",
    "What is resonance and why does it matter in engineering?",
    "How does total internal reflection work in optics?",
    "What is birefringence?",
    "How does a diffraction grating separate light?",
    "What is the difference between coherent and incoherent light?",
    # Astronomy / Space
    "What causes the northern lights?",
    "How do black holes form?",
    "What is a quasar?",
    "What is a binary star system?",
    "What is the difference between a meteor and a meteorite?",
    "How do satellites stay in orbit?",
    "How does sonar detect objects underwater?",
    "What is dark energy and how do we know it exists?",
    "How did the universe form after the Big Bang?",
    "What is a neutron star?",
    "How do astronomers measure the distance to distant galaxies?",
    "What is gravitational lensing?",
    "How does a pulsar emit radiation?",
    "What is the cosmic microwave background?",
    "How do solar flares affect Earth?",
    "What is the Oort cloud?",
    "How does a comet develop its tail?",
    "What is a Lagrange point?",
    "How does aerobraking slow down spacecraft?",
    "What causes a gamma-ray burst?",
    "How does stellar nucleosynthesis produce heavy elements?",
    "What is the Chandrasekhar limit?",
    "How do astronomers detect exoplanets?",
    "What is tidal locking and why is the Moon tidally locked?",
    "How does a space telescope differ from a ground telescope?",
    "What is the difference between a spiral and an elliptical galaxy?",
    "How does the solar wind shape planetary magnetospheres?",
    "What is Hawking radiation?",
    "How does frame dragging work near rotating black holes?",
    "What is a magnetar?",
    # Earth Science / Environment
    "What causes earthquakes?",
    "What is the greenhouse effect?",
    "What is the ozone layer and why does it matter?",
    "What is climate tipping point?",
    "How do tectonic plates move?",
    "What is the water cycle?",
    "What causes tides?",
    "How does carbon capture work?",
    "What is a climate feedback loop?",
    "How does ocean acidification affect marine life?",
    "What is permafrost and why is its thawing alarming?",
    "How do hurricanes form and intensify?",
    "What is the jet stream and how does it affect weather?",
    "How does deforestation affect local rainfall?",
    "What is soil erosion and how can it be prevented?",
    "How does a volcanic eruption affect global temperature?",
    "What is the thermohaline circulation?",
    "How does El Nino affect global weather patterns?",
    "What is groundwater recharge?",
    "How do mangroves protect coastlines?",
    "What is the albedo effect?",
    "How does fracking extract natural gas?",
    "What is seismic tomography?",
    "How does a tsunami form?",
    "What is the difference between weather and climate?",
    "How does the stratosphere protect us from UV radiation?",
    "What is acid rain and what causes it?",
    "How do wildfires affect air quality?",
    "What is microplastic pollution?",
    "How does desalination produce fresh water?",
    # Biology / Medicine
    "Explain how vaccines create immunity.",
    "How do muscles contract?",
    "What is the difference between a virus and a bacterium?",
    "What is DNA and what does it do?",
    "What is CRISPR gene editing?",
    "How does an MRI scanner work?",
    "How do vaccines differ from antibiotics?",
    "How does the brain store memories?",
    "How does the kidney filter blood?",
    "What is mitosis?",
    "What is CRISPR and what can it treat?",
    "How does the liver detoxify the body?",
    "What is a mRNA vaccine?",
    "How does the cochlea convert sound to nerve signals?",
    "How does the human immune system fight infection?",
    "What is antibiotic resistance and why is it dangerous?",
    "How do plants convert sunlight to energy?",
    "What is photosynthesis?",
    "How does the body regulate blood sugar?",
    "What is epigenetics?",
    "How does a stem cell differentiate into a specialised cell?",
    "What is the blood-brain barrier?",
    "How do neurons transmit signals?",
    "What is synaptic plasticity?",
    "How does the lymphatic system work?",
    "What is apoptosis and why is it important?",
    "How do cancer cells evade the immune system?",
    "What is horizontal gene transfer in bacteria?",
    "How does the gut microbiome affect health?",
    "What is RNA interference?",
    "How does the eye focus light?",
    "What is the role of mitochondria beyond energy production?",
    "How does the complement system fight infection?",
    "What is an autoimmune disease?",
    "How do monoclonal antibodies work as drugs?",
    "What is the difference between innate and adaptive immunity?",
    "How does anesthesia work?",
    "What is neuroplasticity?",
    "How does the circadian rhythm regulate the body?",
    "What is prion disease?",
    "How does insulin resistance develop?",
    "What is the mechanism of action of statins?",
    "How do ACE inhibitors lower blood pressure?",
    "What is CAR-T cell therapy?",
    "How does the placenta develop during pregnancy?",
    "What is the role of telomeres in ageing?",
    "How does the spleen filter blood?",
    "What is myelin and what happens when it is damaged?",
    "How does the vestibular system sense balance?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    # Technology / Computing
    "How does Wi-Fi transmit data wirelessly?",
    "How does Bluetooth work?",
    "What is a neural network?",
    "What is the difference between deep learning and machine learning?",
    "How do CPUs execute instructions?",
    "How do radio waves carry information?",
    "How do search engines index the web?",
    "How does a solar panel convert sunlight to electricity?",
    "How do optical fibres carry data?",
    "How do noise-cancelling headphones work?",
    "What is machine learning in simple terms?",
    "How does the internet route data packets?",
    "How does a touchscreen detect your finger?",
    "How does encryption protect data?",
    "How does a wind turbine generate electricity?",
    "How do electric cars differ from petrol cars?",
    "What is a semiconductor?",
    "How does OLED displays produce colour?",
    "How does a gyroscope maintain orientation?",
    "What is a blockchain and how does it work?",
    "How does a compiler turn code into machine instructions?",
    "What is the difference between RISC and CISC architectures?",
    "How does a graphics processing unit differ from a CPU?",
    "What is cache coherence in multiprocessor systems?",
    "How does virtual memory work?",
    "What is a hash function and why is it used in cryptography?",
    "How does public-key cryptography work?",
    "What is a distributed system?",
    "How does a content delivery network reduce latency?",
    "What is the CAP theorem?",
    "How does a database index speed up queries?",
    "What is eventual consistency in distributed databases?",
    "How does a convolutional neural network process images?",
    "What is reinforcement learning?",
    "How does a transformer model process text?",
    "What is attention mechanism in neural networks?",
    "How does backpropagation train a neural network?",
    "What is overfitting and how is it prevented?",
    "How does a generative adversarial network work?",
    "What is federated learning?",
    "How does a recommendation system work?",
    "What is a Kalman filter used for?",
    "How does LiDAR work in autonomous vehicles?",
    "What is SLAM in robotics?",
    "How does a PID controller work?",
    "What is model predictive control?",
    "How does digital signal processing filter noise?",
    "What is a fast Fourier transform used for?",
    "How does lossless compression work?",
    "What is the difference between TCP and UDP?",
    "How does HTTPS secure web traffic?",
    "What is a zero-knowledge proof?",
    "How does a quantum key distribution system work?",
    "What is homomorphic encryption?",
    "How does differential privacy protect data?",
    "What is a side-channel attack?",
    "How does speculative execution cause security vulnerabilities?",
    "What is a hypervisor and how does virtualisation work?",
    "How does containerisation differ from virtualisation?",
    "What is a microservices architecture?",
    "How does a load balancer distribute traffic?",
    "What is a message queue and when is it used?",
    "How does a garbage collector reclaim memory?",
    "What is the difference between stack and heap memory?",
    "How does a JIT compiler improve performance?",
    "What is type inference in programming languages?",
    "How does a relational database enforce ACID properties?",
    "What is an event-driven architecture?",
    "How does a WebSocket differ from HTTP?",
    "What is GraphQL and how does it differ from REST?",
    "How does a MapReduce job process large datasets?",
    "What is a vector database used for?",
    # Chemistry
    "How does a battery store energy?",
    "What is a superconductor?",
    "How does photosynthesis use sunlight to create glucose?",
    "What is Le Chatelier's principle?",
    "How does a catalytic converter reduce emissions?",
    "What is electronegativity and how does it affect bonding?",
    "How do enzymes speed up chemical reactions?",
    "What is the difference between ionic and covalent bonding?",
    "How does osmosis work across a semipermeable membrane?",
    "What is a buffer solution?",
    "How does polymerisation create plastics?",
    "What is the difference between exothermic and endothermic reactions?",
    "How does chromatography separate mixtures?",
    "What is a colloidal suspension?",
    "How does soap remove grease?",
    "What is the role of free radicals in oxidation?",
    "How does titration determine concentration?",
    "What is the Hall-Heroult process for aluminium smelting?",
    "How does galvanisation prevent rust?",
    "What is the Haber-Bosch process?",
    "How does thermite work?",
    "What is a noble gas and why is it unreactive?",
    "How does nuclear magnetic resonance spectroscopy identify molecules?",
    "What is chirality in chemistry?",
    "How does a fuel cell produce electricity from hydrogen?",
    "What is the difference between distillation and fractional distillation?",
    "How does reverse osmosis purify water?",
    "What is a zeolite used for?",
    "How does fluorine form such strong bonds?",
    "What is the Maillard reaction in cooking?",
    # Engineering / Materials
    "How does a jet engine produce thrust?",
    "How do aeroplanes generate lift?",
    "How does an MRI scanner work?",
    "How do electric cars differ from petrol cars?",
    "How does a wind turbine generate electricity?",
    "How does a heat pump work?",
    "How does a nuclear reactor generate electricity?",
    "How does sonar detect objects underwater?",
    "What is the difference between tensile and compressive strength?",
    "How does carbon fibre achieve such high strength-to-weight ratio?",
    "What is work hardening in metals?",
    "How does annealing change material properties?",
    "What is the difference between steel and cast iron?",
    "How does 3D printing layer materials to create objects?",
    "What is selective laser sintering?",
    "How does a turbocharger work?",
    "What is regenerative braking?",
    "How does active noise control work in aircraft cabins?",
    "What is fatigue failure in engineering?",
    "How does a piezoelectric actuator work?",
    "What is the finite element method?",
    "How does a flywheel store energy?",
    "What is cavitation in pumps?",
    "How does a heat exchanger transfer thermal energy?",
    "What is tribology and why does it matter?",
    "How does hydraulic fracturing extract oil?",
    "What is corrosion and how is it prevented?",
    "How does a bridge distribute load forces?",
    "What is prestressed concrete?",
    "How does a reciprocating compressor work?",
    "What is the difference between laminar and turbulent flow?",
    "How does a venturi meter measure flow rate?",
    "What is boundary layer separation in aerodynamics?",
    "How does a regenerative heat exchanger work?",
    "What is the Rankine cycle in power plants?",
    # Economics / Finance / Society
    "How does the stock market determine prices?",
    "What is inflation and what causes it?",
    "How does a central bank control money supply?",
    "What is quantitative easing?",
    "How does fractional reserve banking work?",
    "What is a bond yield and how does it relate to price?",
    "How do options and futures contracts work?",
    "What is the difference between GDP and GNP?",
    "How does a currency peg work?",
    "What is comparative advantage in international trade?",
    "How does a credit default swap work?",
    "What is moral hazard in finance?",
    "How does insider trading harm markets?",
    "What is a Ponzi scheme?",
    "How do central banks set interest rates?",
    "What is the Phillips curve?",
    "How does rent control affect housing supply?",
    "What is the prisoner's dilemma?",
    "How do externalities affect market outcomes?",
    "What is the tragedy of the commons?",
    "How does the Gini coefficient measure inequality?",
    "What is purchasing power parity?",
    "How does a progressive tax system work?",
    "What is modern monetary theory?",
    "How do trade tariffs affect domestic industries?",
    # Psychology / Neuroscience
    "How does the brain store memories?",
    "What is neuroplasticity?",
    "How does dopamine affect motivation?",
    "What is the difference between working memory and long-term memory?",
    "How does sleep consolidate memory?",
    "What is confirmation bias?",
    "How does cognitive dissonance affect decision-making?",
    "What is the placebo effect?",
    "How does stress affect the hippocampus?",
    "What is mirror neuron theory?",
    "How does addiction alter brain chemistry?",
    "What is the default mode network?",
    "How does deep brain stimulation treat Parkinson's disease?",
    "What is the difference between classical and operant conditioning?",
    "How does meditation change brain structure?",
    "What is prosopagnosia?",
    "How does the amygdala process fear?",
    "What is synaesthesia?",
    "How does bilingualism affect cognitive reserve?",
    "What is the cocktail party effect in auditory processing?",
    # History / Philosophy
    "What were the main causes of World War One?",
    "How did the printing press change European society?",
    "What was the significance of the Magna Carta?",
    "How did the Industrial Revolution transform living conditions?",
    "What caused the fall of the Roman Empire?",
    "How did the Black Death reshape medieval Europe?",
    "What were the causes of the French Revolution?",
    "How did colonialism shape the modern world economy?",
    "What was the significance of the Manhattan Project?",
    "How did the Cold War shape global politics?",
    "What is the trolley problem in ethics?",
    "How does Kant's categorical imperative work?",
    "What is utilitarianism?",
    "What is the ship of Theseus paradox?",
    "How does John Rawls justify social justice?",
    "What is the difference between deductive and inductive reasoning?",
    "What is Occam's razor?",
    "What is the Chinese room argument against AI consciousness?",
    "How did Darwin develop the theory of natural selection?",
    "What is the difference between empiricism and rationalism?",
    # Mathematics (no LaTeX notation)
    "What is the Pythagorean theorem and how is it proved?",
    "Why is the number pi irrational?",
    "What is a prime number and how are large primes found?",
    "How does the RSA algorithm use prime numbers for encryption?",
    "What is a Fourier series?",
    "What is the difference between permutations and combinations?",
    "How does Bayes theorem update probabilities with new evidence?",
    "What is the central limit theorem?",
    "What is a Markov chain?",
    "How does gradient descent find the minimum of a function?",
    "What is the difference between a vector and a matrix?",
    "What is an eigenvalue used for?",
    "What is a partial differential equation?",
    "What is the travelling salesman problem?",
    "What is a graph in mathematics and what is it used for?",
    "What is the difference between NP and NP-hard problems?",
    "How does the simplex method solve linear programs?",
    "What is a Monte Carlo simulation?",
    "What is the law of large numbers?",
    "What is a convex function and why does it matter in optimisation?",
    # Energy / Sustainability
    "How does a solar panel convert sunlight to electricity?",
    "How does a wind turbine generate electricity?",
    "How does a hydroelectric dam generate power?",
    "What is tidal energy and how is it harnessed?",
    "How does geothermal energy work?",
    "What is the capacity factor of a power plant?",
    "How does energy storage in lithium-ion batteries work?",
    "What is pumped hydro storage?",
    "How does a smart grid manage electricity demand?",
    "What is demand response in electricity markets?",
    "How does a combined cycle gas plant improve efficiency?",
    "What is carbon intensity of electricity generation?",
    "How do green hydrogen electrolysers work?",
    "What is carbon-neutral versus carbon-negative?",
    "How does life cycle analysis assess environmental impact?",
    "What is the energy return on energy invested ratio?",
    "How do offshore wind farms anchor to the seabed?",
    "What is agrivoltaics?",
    "How does district heating reduce energy waste?",
    "What is a virtual power plant?",
    # Food / Life Science
    "Why does bread rise when baked?",
    "How does fermentation produce alcohol?",
    "What is the role of gluten in bread texture?",
    "How does pasteurisation preserve food?",
    "What is the difference between saturated and unsaturated fats?",
    "How does the body digest proteins?",
    "What is the glycaemic index?",
    "How do probiotics benefit gut health?",
    "What is umami and why does it taste savoury?",
    "How does caffeine reduce tiredness?",
    "What is the difference between soluble and insoluble fibre?",
    "How does the liver process alcohol?",
    "What is the role of iron in the blood?",
    "How does vitamin D deficiency affect bone health?",
    "What is the mechanism behind food allergies?",
    # Language / Linguistics
    "How do children acquire language so rapidly?",
    "What is the Sapir-Whorf hypothesis?",
    "How do tonal languages like Mandarin differ from English?",
    "What is the difference between syntax and semantics?",
    "How does sign language convey grammar?",
    "What is a pidgin language?",
    "How did Proto-Indo-European language spread across continents?",
    "What is code-switching in bilingual speakers?",
    "How does the brain process reading differently from speech?",
    "What is linguistic relativity?",
    # Architecture / Materials Science
    "How does a suspension bridge distribute loads?",
    "What makes concrete strong and why does it crack?",
    "How does a geodesic dome achieve structural efficiency?",
    "What is passive solar design in architecture?",
    "How do earthquake-resistant buildings absorb seismic energy?",
    "What is a smart material?",
    "How does graphene compare to steel in strength?",
    "What is aerogel and what makes it such a good insulator?",
    "How does self-healing concrete work?",
    "What is the difference between hardness and toughness in materials?",
    # Miscellaneous interesting questions
    "Why does the sky appear blue?",
    "How do migratory birds navigate thousands of kilometres?",
    "Why do we dream and what is REM sleep?",
    "How does a dog's sense of smell work?",
    "Why do metals conduct electricity but most plastics do not?",
    "How do social insects like ants coordinate without a leader?",
    "Why does hot water sometimes freeze faster than cold water?",
    "How does colour vision work in the human eye?",
    "Why do some materials glow in the dark?",
    "How does a soap bubble stay spherical?",
    "Why is glass transparent?",
    "How do bats navigate using echolocation?",
    "Why does ice float on water?",
    "How do plants know which direction is up?",
    "Why do we get goosebumps?",
    "How do cephalopods like octopuses change colour so rapidly?",
    "Why do fingers wrinkle in water?",
    "How does a Venus flytrap detect insects?",
    "Why does iron rust but gold does not?",
    "How do homing pigeons find their way home?",
]


def make_benchmark_prompt(index, repeat):
    seeds = [
        "Explain queueing delay and model service time in one concise paragraph.",
        "Summarize how admission control changes latency and throughput.",
        "Describe why GPU power can change under different request schedules.",
        "Compare eager scheduling with controlled queue wait for LLM serving.",
    ]
    return " ".join([seeds[index % len(seeds)]] * max(1, repeat))


def run_internal_budget_sweep(body):
    fractions = [float(x) for x in body.get("admission_fractions", [1.0, 0.75, 0.5, 0.25, 0.1, 0.05])]
    offered_rate = float(body.get("offered_rate_qps", 4.0))
    duration_s = float(body.get("duration_s", 30.0))
    warmup_s = float(body.get("warmup_s", 5.0))
    max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
    prompt_repeat = int(body.get("prompt_repeat", 48))
    seed = int(body.get("seed", 10))
    random.seed(seed)

    summaries = []
    for fraction in fractions:
        fraction = max(0.01, min(1.0, fraction))
        control = {
            "mode": "open_loop",
            "admission_fraction": fraction,
            "enabled": True,
            "source": "run_internal_budget_sweep",
            "timestamp": datetime.now().isoformat(),
        }
        write_scheduler_control(control)
        log(f"internal budget sweep set scheduler control {CONTROL_FILE}: {control}")
        time.sleep(float(body.get("settle_s", 2.0)))

        before = fetch_backend_metrics()
        stop = threading.Event()
        sem = threading.Semaphore(int(body.get("max_outstanding", 256)))
        lock = threading.Lock()
        records = []
        power_samples = []

        def power_loop():
            while not stop.is_set():
                sample = gpu_snapshot()
                sample["t"] = time.perf_counter()
                power_samples.append(sample)
                time.sleep(float(body.get("metric_period_s", 1.0)))

        def one_request(i, measure):
            prompt = make_benchmark_prompt(i, prompt_repeat)
            t_send = time.perf_counter()
            t_first = None
            status = "ok"
            try:
                payload = {
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                }
                with requests.post(
                    f"{BACKEND_URL}/v1/completions",
                    data=json.dumps(payload),
                    headers=headers(),
                    stream=True,
                    timeout=TIMEOUT,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_lines():
                        if chunk and chunk != b"data: [DONE]" and t_first is None:
                            t_first = time.perf_counter()
            except Exception as exc:
                status = f"error:{exc!r}"
            finally:
                t_done = time.perf_counter()
                if measure:
                    with lock:
                        records.append(
                            {
                                "status": status,
                                "ttft_ms": 1000.0 * (t_first - t_send) if t_first else None,
                                "total_ms": 1000.0 * (t_done - t_send),
                            }
                        )
                sem.release()

        power_thread = threading.Thread(target=power_loop, daemon=True)
        power_thread.start()
        threads = []
        t_start = time.perf_counter()
        t_measure_start = t_start + warmup_s
        t_end = t_measure_start + duration_s
        next_arrival = t_start
        req_id = 0
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now < next_arrival:
                time.sleep(min(0.01, next_arrival - now))
                continue
            if sem.acquire(timeout=0.1):
                req_id += 1
                measure = now >= t_measure_start
                thread = threading.Thread(target=one_request, args=(req_id, measure), daemon=True)
                thread.start()
                threads.append(thread)
            next_arrival += random.expovariate(offered_rate) if offered_rate > 0 else 1.0

        for thread in threads:
            thread.join(timeout=TIMEOUT)
        stop.set()
        power_thread.join(timeout=5)
        after = fetch_backend_metrics()

        ok = [r for r in records if r["status"] == "ok"]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in ok]
        energy = integrate_power(power_samples)
        summaries.append(
            {
                "admission_fraction": fraction,
                "control": control,
                "offered_rate_qps": offered_rate,
                "requests_measured": len(records),
                "requests_ok": len(ok),
                "error_rate": 1.0 - len(ok) / max(len(records), 1),
                "throughput_req_s": len(ok) / max(duration_s, 1e-9),
                "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
                "ttft_p95_ms": percentile(ttfts, 95),
                "total_mean_ms": statistics.mean(totals) if totals else None,
                "total_p95_ms": percentile(totals, 95),
                "vllm_queue_wait_mean_ms": metric_delta_mean_ms(before, after, "vllm:request_queue_time_seconds"),
                "vllm_ttft_mean_ms": metric_delta_mean_ms(before, after, "vllm:time_to_first_token_seconds"),
                "vllm_e2e_mean_ms": metric_delta_mean_ms(before, after, "vllm:e2e_request_latency_seconds"),
                "gpu_power_mean_w": statistics.mean(
                    [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x]
                )
                if any("gpu_power_w" in x for x in power_samples)
                else None,
                "gpu_power_peak_w": max(
                    [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x],
                    default=None,
                ),
                "energy_j": energy,
                "energy_per_request_j": energy / len(ok) if energy is not None and ok else None,
            }
        )
    return {"status": "ok", "summaries": summaries}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def run_internal_ttft_sweep(body):
    """Closed-loop TTFT controller sweep (Phase 2).

    Supports two actuator modes selected by the 'actuator' field:

    'token_budget' (original): writes measured TTFT to the scheduler control
    file; the vLLM ControlledScheduler PI adjusts admission_fraction.
    Limitation: the queue must already be loaded for the actuator to raise
    TTFT above natural; not suitable for stable set-point tracking at light
    load.

    'dispatch_delay' (default): the wrapper sleeps for dispatch_delay_ms
    before sending each request to vLLM, then updates the delay via PI.
    Plant gain is positive (delay↑ → TTFT↑), so TTFT = natural + delay.
    This gives direct, stable set-point tracking independent of queue state.
    """
    actuator = str(body.get("actuator", "dispatch_delay"))
    targets = [float(x) for x in body.get("target_ttft_ms", [200.0, 300.0])]
    offered_rate = float(body.get("offered_rate_qps", 4.0))
    duration_s = float(body.get("duration_s", 60.0))
    warmup_s = float(body.get("warmup_s", 10.0))
    settle_s = float(body.get("settle_s", 3.0))
    max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
    prompt_repeat = int(body.get("prompt_repeat", 64))
    feedback_period_s = float(body.get("feedback_period_s", 0.5))
    ttft_window = int(body.get("ttft_window", 20))
    kp = float(body.get("kp", 0.15))
    ki = float(body.get("ki", 0.02))
    fraction_min = float(body.get("fraction_min", 0.25))
    fraction_max = float(body.get("fraction_max", 1.0))
    max_delay_ms = float(body.get("max_delay_ms", 5000.0))
    seed = int(body.get("seed", 10))
    random.seed(seed)

    all_results = []
    for target_ttft in targets:
        # Dispatch-delay state (shared via list for closure mutability).
        dispatch_delay_ms = [0.0]
        delay_xi = [0.0]

        if actuator == "dispatch_delay":
            # Scheduler runs open-loop at full budget; all TTFT shaping is
            # done by the explicit per-request delay inserted in one_request().
            initial_ctrl = {
                "mode": "open_loop",
                "admission_fraction": 1.0,
                "enabled": True,
                "source": "run_internal_ttft_sweep_delay",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            initial_ctrl = {
                "mode": "ttft",
                "target_ttft_ms": target_ttft,
                "measured_ttft_ms": None,
                "enabled": True,
                "kp": kp,
                "ki": ki,
                "fraction_min": fraction_min,
                "fraction_max": fraction_max,
                "admission_fraction": fraction_max,
                "source": "run_internal_ttft_sweep",
                "timestamp": datetime.now().isoformat(),
            }
        write_scheduler_control(initial_ctrl)
        log(f"ttft_sweep: actuator={actuator} target={target_ttft} ms kp={kp} ki={ki} settle={settle_s}s")
        time.sleep(settle_s)

        before = fetch_backend_metrics()
        stop = threading.Event()
        sem = threading.Semaphore(int(body.get("max_outstanding", 256)))
        lock = threading.Lock()
        records = []
        power_samples = []
        timeseries = []
        recent_ttfts = collections.deque(maxlen=ttft_window)
        sample_errors: list[str] = []

        t_start = time.perf_counter()
        t_measure_start = t_start + warmup_s
        t_end = t_measure_start + duration_s

        def check_backend_alive():
            try:
                resp = requests.get(HEALTH_URL, timeout=3.0)
                return resp.ok
            except Exception:
                return False

        def feedback_loop():
            while not stop.is_set():
                if not check_backend_alive():
                    log(f"ttft_sweep: vLLM backend at {HEALTH_URL} is DOWN — stopping sweep")
                    stop.set()
                    return
                with lock:
                    window = list(recent_ttfts)
                measured = statistics.mean(window) if window else None

                if actuator == "dispatch_delay" and measured is not None:
                    # Positive-gain plant: delay↑ → TTFT↑.
                    # Standard PI: e > 0 when measured < target → increase delay.
                    e_norm = (target_ttft - measured) / max(target_ttft, 1.0)
                    dt = feedback_period_s
                    at_floor = dispatch_delay_ms[0] <= 1e-6
                    at_ceil = dispatch_delay_ms[0] >= max_delay_ms - 1e-6
                    should_integrate = not (at_floor and e_norm < 0) and not (at_ceil and e_norm > 0)
                    if should_integrate:
                        delay_xi[0] = _clamp(delay_xi[0] + e_norm * dt, -20.0, 20.0)
                    # Scale by target so kp/ki are dimensionless and comparable
                    # to the token-budget mode: delta in ms.
                    delta_ms = (kp * e_norm + ki * delay_xi[0]) * target_ttft
                    dispatch_delay_ms[0] = _clamp(dispatch_delay_ms[0] + delta_ms, 0.0, max_delay_ms)
                    log(
                        "ttft_sweep delay_pi target=%.1f measured=%.1f e=%.3f xi=%.3f delay=%.1fms delta=%.2f",
                        target_ttft,
                        measured,
                        e_norm,
                        delay_xi[0],
                        dispatch_delay_ms[0],
                        delta_ms,
                    ) if False else None  # verbose only; remove 'if False' to enable

                sched = scheduler_status() if actuator != "dispatch_delay" else {}
                gpu = gpu_snapshot()
                now = time.perf_counter()
                timeseries.append({
                    "t": round(now - t_start, 3),
                    "target_ttft_ms": target_ttft,
                    "measured_ttft_ms": round(measured, 2) if measured is not None else None,
                    "actuator": actuator,
                    "dispatch_delay_ms": round(dispatch_delay_ms[0], 2),
                    "delay_xi": round(delay_xi[0], 4),
                    "admission_fraction": sched.get("scheduler_admission_fraction"),
                    "token_cap": sched.get("scheduler_token_cap"),
                    "xi": sched.get("scheduler_xi"),
                    "gpu_power_w": gpu.get("gpu_power_w"),
                })

                if actuator != "dispatch_delay":
                    ctrl = {
                        "mode": "ttft",
                        "target_ttft_ms": target_ttft,
                        "measured_ttft_ms": measured,
                        "enabled": True,
                        "kp": kp,
                        "ki": ki,
                        "fraction_min": fraction_min,
                        "fraction_max": fraction_max,
                        "source": "feedback_loop",
                        "timestamp": datetime.now().isoformat(),
                    }
                    write_scheduler_control(ctrl)
                time.sleep(feedback_period_s)

        def power_loop():
            while not stop.is_set():
                sample = gpu_snapshot()
                sample["t"] = time.perf_counter()
                power_samples.append(sample)
                time.sleep(float(body.get("metric_period_s", 0.5)))

        def one_request(i, measure):
            prompt = make_benchmark_prompt(i, prompt_repeat)
            # t_send is captured BEFORE the dispatch delay so the delay is
            # included in measured TTFT (client experiences delay + prefill).
            t_send = time.perf_counter()
            with lock:
                delay_s = dispatch_delay_ms[0] / 1000.0
            if delay_s > 0:
                time.sleep(delay_s)
            t_first = None
            status = "ok"
            try:
                payload = {
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                }
                with requests.post(
                    f"{BACKEND_URL}/v1/completions",
                    data=json.dumps(payload),
                    headers=headers(),
                    stream=True,
                    timeout=TIMEOUT,
                ) as resp:
                    resp.raise_for_status()
                    for chunk in resp.iter_lines():
                        if chunk and chunk != b"data: [DONE]" and t_first is None:
                            t_first = time.perf_counter()
            except Exception as exc:
                status = f"error:{exc!r}"
                with lock:
                    if len(sample_errors) < 5:
                        sample_errors.append(repr(exc))
            finally:
                t_done = time.perf_counter()
                if t_first is not None:
                    ttft_ms = 1000.0 * (t_first - t_send)
                    with lock:
                        recent_ttfts.append(ttft_ms)
                if measure:
                    with lock:
                        records.append({
                            "status": status,
                            "ttft_ms": 1000.0 * (t_first - t_send) if t_first else None,
                            "total_ms": 1000.0 * (t_done - t_send),
                        })
                sem.release()

        feedback_thread = threading.Thread(target=feedback_loop, daemon=True)
        power_thread = threading.Thread(target=power_loop, daemon=True)
        feedback_thread.start()
        power_thread.start()

        threads = []
        next_arrival = t_start
        req_id = 0
        while time.perf_counter() < t_end:
            now = time.perf_counter()
            if now < next_arrival:
                time.sleep(min(0.01, next_arrival - now))
                continue
            if sem.acquire(timeout=0.1):
                req_id += 1
                measure = now >= t_measure_start
                thread = threading.Thread(target=one_request, args=(req_id, measure), daemon=True)
                thread.start()
                threads.append(thread)
            next_arrival += random.expovariate(offered_rate) if offered_rate > 0 else 1.0

        for thread in threads:
            thread.join(timeout=TIMEOUT)
        stop.set()
        feedback_thread.join(timeout=5)
        power_thread.join(timeout=5)
        after = fetch_backend_metrics()

        ok = [r for r in records if r["status"] == "ok"]
        ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
        totals = [r["total_ms"] for r in ok]
        energy = integrate_power(power_samples)

        all_results.append({
            "target_ttft_ms": target_ttft,
            "offered_rate_qps": offered_rate,
            "kp": kp,
            "ki": ki,
            "fraction_min": fraction_min,
            "fraction_max": fraction_max,
            "requests_measured": len(records),
            "requests_ok": len(ok),
            "error_rate": 1.0 - len(ok) / max(len(records), 1),
            "throughput_req_s": len(ok) / max(duration_s, 1e-9),
            "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
            "ttft_p95_ms": percentile(ttfts, 95),
            "total_mean_ms": statistics.mean(totals) if totals else None,
            "total_p95_ms": percentile(totals, 95),
            "vllm_queue_wait_mean_ms": metric_delta_mean_ms(before, after, "vllm:request_queue_time_seconds"),
            "vllm_ttft_mean_ms": metric_delta_mean_ms(before, after, "vllm:time_to_first_token_seconds"),
            "vllm_e2e_mean_ms": metric_delta_mean_ms(before, after, "vllm:e2e_request_latency_seconds"),
            "gpu_power_mean_w": statistics.mean(
                [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x]
            ) if any("gpu_power_w" in x for x in power_samples) else None,
            "gpu_power_peak_w": max(
                [float(x["gpu_power_w"]) for x in power_samples if "gpu_power_w" in x],
                default=None,
            ),
            "energy_j": energy,
            "energy_per_request_j": energy / len(ok) if energy is not None and ok else None,
            "sample_errors": sample_errors[:5],
            "timeseries": timeseries,
        })

    write_scheduler_control({
        "mode": "open_loop",
        "admission_fraction": 1.0,
        "enabled": True,
        "source": "ttft_sweep_done",
        "timestamp": datetime.now().isoformat(),
    })

    return {"status": "ok", "results": all_results}


def run_internal_load_step(body):
    """Disturbance rejection: fixed TTFT target(s) with stepping offered load.

    When target_ttft_ms is a list, each target is run sequentially and all
    results are returned together so the client makes only one HTTP call.
    """
    targets_raw = body.get("target_ttft_ms", 300.0)
    all_targets = ([float(x) for x in targets_raw]
                   if isinstance(targets_raw, list) else [float(targets_raw)])

    actuator = str(body.get("actuator", "token_budget"))
    # Mutable cell so feedback_loop sees target updates across target transitions.
    target_ttft = [all_targets[0]]
    load_steps = list(body.get("load_steps", [
        {"qps": 2.0, "duration_s": 90.0},
        {"qps": 4.0, "duration_s": 90.0},
        {"qps": 2.0, "duration_s": 90.0},
    ]))
    warmup_qps = float(body.get("warmup_qps", 2.0))
    warmup_s = float(body.get("warmup_s", 90.0))
    warmup_fraction = float(body.get("warmup_fraction", 0.08))
    kp = float(body.get("kp", 0.05))
    ki = float(body.get("ki", 0.01))
    fraction_min = float(body.get("fraction_min", 0.05))
    fraction_max = float(body.get("fraction_max", 1.0))
    max_delay_ms = float(body.get("max_delay_ms", 2000.0))
    ttft_window = int(body.get("ttft_window", 20))
    feedback_period_s = float(body.get("feedback_period_s", 0.5))
    max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
    prompt_repeat = int(body.get("prompt_repeat", 64))
    seed = int(body.get("seed", 10))
    random.seed(seed)

    # Mutable shared state (lists for closure capture).
    phase_label = ["warmup"]
    offered_qps = [warmup_qps]
    dispatch_delay_ms = [0.0]
    delay_xi = [0.0]

    stop = threading.Event()
    sem = threading.Semaphore(int(body.get("max_outstanding", 256)))
    lock = threading.Lock()
    records = []
    power_samples = []
    timeseries = []
    qa_log = []
    recent_ttfts = collections.deque(maxlen=ttft_window)
    sample_errors = []

    t_start = time.perf_counter()

    def check_alive():
        try:
            return requests.get(HEALTH_URL, timeout=3.0).ok
        except Exception:
            return False

    def power_loop():
        while not stop.is_set():
            sample = gpu_snapshot()
            sample["t"] = time.perf_counter()
            power_samples.append(sample)
            time.sleep(float(body.get("metric_period_s", 0.5)))

    _last_alive_check = [0.0]
    _alive_interval_s = 5.0  # don't block fast feedback loops with HTTP health checks

    def feedback_loop():
        while not stop.is_set():
            now_alive = time.perf_counter()
            if now_alive - _last_alive_check[0] >= _alive_interval_s:
                if not check_alive():
                    log("load_step: vLLM backend down — stopping")
                    stop.set()
                    return
                _last_alive_check[0] = now_alive
            with lock:
                window = list(recent_ttfts)
                cur_phase = phase_label[0]
                cur_qps = offered_qps[0]
            measured = statistics.mean(window) if window else None

            if cur_phase != "warmup" and measured is not None:
                if actuator == "dispatch_delay":
                    e_norm = (target_ttft[0] - measured) / max(target_ttft[0], 1.0)
                    at_floor = dispatch_delay_ms[0] <= 1e-6
                    at_ceil = dispatch_delay_ms[0] >= max_delay_ms - 1e-6
                    should_integrate = not (at_floor and e_norm < 0) and not (at_ceil and e_norm > 0)
                    if should_integrate:
                        delay_xi[0] = _clamp(delay_xi[0] + e_norm * feedback_period_s, -20.0, 20.0)
                    delta_ms = (kp * e_norm + ki * delay_xi[0]) * target_ttft[0]
                    dispatch_delay_ms[0] = _clamp(dispatch_delay_ms[0] + delta_ms, 0.0, max_delay_ms)
                else:
                    write_scheduler_control({
                        "mode": "ttft",
                        "target_ttft_ms": target_ttft[0],
                        "measured_ttft_ms": measured,
                        "enabled": True,
                        "kp": kp,
                        "ki": ki,
                        "fraction_min": fraction_min,
                        "fraction_max": fraction_max,
                        "source": "load_step_feedback",
                        "timestamp": datetime.now().isoformat(),
                    })

            sched = scheduler_status() if actuator != "dispatch_delay" else {}
            gpu = gpu_snapshot()
            now = time.perf_counter()
            timeseries.append({
                "t": round(now - t_start, 3),
                "phase": cur_phase,
                "offered_qps": cur_qps,
                "target_ttft_ms": target_ttft[0],
                "measured_ttft_ms": round(measured, 2) if measured is not None else None,
                "actuator": actuator,
                "dispatch_delay_ms": round(dispatch_delay_ms[0], 2),
                "delay_xi": round(delay_xi[0], 4),
                "admission_fraction": sched.get("scheduler_admission_fraction"),
                "token_cap": sched.get("scheduler_token_cap"),
                "xi": sched.get("scheduler_xi"),
                "gpu_power_w": gpu.get("gpu_power_w"),
                "gpu_util_percent": gpu.get("gpu_util_percent"),
            })
            time.sleep(feedback_period_s)

    def one_request(i, measure):
        question = QUESTION_BANK[i % len(QUESTION_BANK)]
        # t_send captured BEFORE delay so TTFT includes dispatch delay.
        t_send = time.perf_counter()
        sent_at_s = t_send - t_start
        with lock:
            delay_s = dispatch_delay_ms[0] / 1000.0
            cur_phase = phase_label[0]
            cur_qps = offered_qps[0]
        if delay_s > 0:
            time.sleep(delay_s)
        t_first = None
        answer_parts: list[str] = []
        status = "ok"
        try:
            payload = {
                "model": MODEL,
                "prompt": question,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
            }
            with requests.post(
                f"{BACKEND_URL}/v1/completions",
                data=json.dumps(payload),
                headers=headers(),
                stream=True,
                timeout=TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_lines():
                    if not chunk or chunk == b"data: [DONE]":
                        continue
                    if t_first is None:
                        t_first = time.perf_counter()
                    try:
                        raw = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
                        if raw.startswith("data: "):
                            raw = raw[6:]
                        token_text = json.loads(raw)["choices"][0].get("text", "")
                        answer_parts.append(token_text)
                    except Exception:
                        pass
        except Exception as exc:
            status = f"error:{exc!r}"
            with lock:
                if len(sample_errors) < 5:
                    sample_errors.append(repr(exc))
        finally:
            t_done = time.perf_counter()
            ttft_ms = 1000.0 * (t_first - t_send) if t_first else None
            recv_at_s = t_done - t_start
            if ttft_ms is not None:
                with lock:
                    recent_ttfts.append(ttft_ms)
            if measure:
                with lock:
                    records.append({
                        "status": status,
                        "ttft_ms": ttft_ms,
                        "total_ms": 1000.0 * (t_done - t_send),
                        "phase": cur_phase,
                    })
                    qa_log.append({
                        "question": question,
                        "answer": "".join(answer_parts).strip(),
                        "sent_at_s": round(sent_at_s, 3),
                        "recv_at_s": round(recv_at_s, 3),
                        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
                        "total_ms": round(1000.0 * (t_done - t_send), 1),
                        "phase": cur_phase,
                        "offered_qps": cur_qps,
                    })
            sem.release()

    # Set initial scheduler state for warmup phase.
    write_scheduler_control({
        "mode": "open_loop",
        "admission_fraction": warmup_fraction if actuator == "token_budget" else 1.0,
        "enabled": True,
        "source": "load_step_warmup",
        "timestamp": datetime.now().isoformat(),
    })

    power_thread = threading.Thread(target=power_loop, daemon=True)
    feedback_thread = threading.Thread(target=feedback_loop, daemon=True)
    power_thread.start()
    feedback_thread.start()

    def dispatch_phase(qps, duration_s, measure):
        t_end = time.perf_counter() + duration_s
        next_arrival = time.perf_counter()
        idx = 0
        threads = []
        while time.perf_counter() < t_end and not stop.is_set():
            now = time.perf_counter()
            if now < next_arrival:
                time.sleep(min(0.01, next_arrival - now))
                continue
            if sem.acquire(timeout=0.1):
                idx += 1
                t = threading.Thread(target=one_request, args=(idx, measure), daemon=True)
                t.start()
                threads.append(t)
            next_arrival += random.expovariate(qps) if qps > 0 else 1.0
        for t in threads:
            t.join(timeout=TIMEOUT)

    # Phase 1: warmup (open-loop, not measured).
    log(f"load_step: warmup qps={warmup_qps} dur={warmup_s}s actuator={actuator} "
        f"fraction={warmup_fraction if actuator == 'token_budget' else 1.0}")
    with lock:
        phase_label[0] = "warmup"
        offered_qps[0] = warmup_qps
    dispatch_phase(warmup_qps, warmup_s, measure=False)
    log("load_step: warmup complete — switching to closed-loop")

    # Switch to closed-loop. For token_budget, scheduler starts PI from the
    # warmup equilibrium fraction (not reset to 1.0) so the transition is smooth.
    # The scheduler resets _xi on mode change but preserves _admission_fraction.
    if actuator == "token_budget":
        write_scheduler_control({
            "mode": "ttft",
            "target_ttft_ms": target_ttft[0],
            "measured_ttft_ms": None,
            "enabled": True,
            "kp": kp,
            "ki": ki,
            "fraction_min": fraction_min,
            "fraction_max": fraction_max,
            "source": "load_step_cl_start",
            "timestamp": datetime.now().isoformat(),
        })

    # Phase 2: closed-loop load steps — cycle through all targets sequentially.
    # Controller state (delay_ms, delay_xi) carries over across target transitions
    # so the timeseries shows the live re-settling transient.
    for tgt_idx, tgt in enumerate(all_targets):
        with lock:
            target_ttft[0] = tgt
            recent_ttfts.clear()   # flush stale window so new target sees fresh measurements
        log(f"load_step: → target {tgt:.0f}ms ({tgt_idx + 1}/{len(all_targets)})")
        for i, step in enumerate(load_steps):
            qps = float(step["qps"])
            dur = float(step["duration_s"])
            label = f"step_{i}_qps{qps:.0f}_t{int(tgt)}ms"
            log(f"load_step: {label} qps={qps} dur={dur}s")
            with lock:
                phase_label[0] = label
                offered_qps[0] = qps
            dispatch_phase(qps, dur, measure=True)

    stop.set()
    feedback_thread.join(timeout=5)
    power_thread.join(timeout=5)

    write_scheduler_control({
        "mode": "open_loop",
        "admission_fraction": 1.0,
        "enabled": True,
        "source": "load_step_done",
        "timestamp": datetime.now().isoformat(),
    })

    ok = [r for r in records if r["status"] == "ok"]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    energy = integrate_power(power_samples)

    step_summaries = []
    for tgt in all_targets:
        for i, step in enumerate(load_steps):
            label = f"step_{i}_qps{float(step['qps']):.0f}_t{int(tgt)}ms"
            step_ok = [r for r in records if r.get("phase") == label and r["status"] == "ok"]
            step_ttfts = [r["ttft_ms"] for r in step_ok if r["ttft_ms"] is not None]
            step_summaries.append({
                "target_ttft_ms": tgt,
                "step": i,
                "qps": step["qps"],
                "duration_s": step["duration_s"],
                "requests_ok": len(step_ok),
                "ttft_mean_ms": statistics.mean(step_ttfts) if step_ttfts else None,
                "ttft_p95_ms": percentile(step_ttfts, 95),
                "ttft_stdev_ms": statistics.stdev(step_ttfts) if len(step_ttfts) > 1 else None,
            })

    return {
        "status": "ok",
        "target_ttft_ms": all_targets,
        "actuator": actuator,
        "warmup_qps": warmup_qps,
        "warmup_s": warmup_s,
        "warmup_fraction": warmup_fraction,
        "load_steps": load_steps,
        "requests_measured": len(records),
        "requests_ok": len(ok),
        "error_rate": 1.0 - len(ok) / max(len(records), 1),
        "ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
        "ttft_p95_ms": percentile(ttfts, 95),
        "energy_j": energy,
        "sample_errors": sample_errors[:5],
        "step_summaries": step_summaries,
        "timeseries": timeseries,
        "qa_log": qa_log,
    }


def recent_arrival_rate():
    now = time.perf_counter()
    recent = [t for t in ARRIVAL_TS if now - t <= 10.0]
    if not recent:
        return 0.0
    return round(len(recent) / 10.0, 2)


def update_queue_area_locked(now=None):
    global QUEUE_AREA, QUEUE_LAST_TS
    if now is None:
        now = time.perf_counter()
    dt = now - QUEUE_LAST_TS
    if dt > 0:
        QUEUE_AREA += len(FIFO) * dt
        QUEUE_LAST_TS = now
    return now


def new_request_id():
    global REQ_COUNTER
    with LOCK:
        REQ_COUNTER += 1
        return f"r{REQ_COUNTER:06d}"


def build_queue_item(prompt, prompt_repeat, max_tokens, temperature, source, client_ts):
    expanded_prompt = prompt if prompt_repeat <= 1 else (prompt + " ") * prompt_repeat
    request_id = new_request_id()
    return QueueItem(
        request_id=request_id,
        prompt=expanded_prompt,
        prompt_chars=len(expanded_prompt),
        prompt_repeat=prompt_repeat,
        max_tokens=max_tokens,
        temperature=temperature,
        source=source,
        client_ts=client_ts,
        enqueued_wall=datetime.now().isoformat(),
        enqueued_perf=time.perf_counter(),
    )


def enqueue_item(item):
    global TICK_ARRIVALS, TICK_Q_MAX
    with LOCK:
        now = update_queue_area_locked()
        FIFO.append(item)
        ARRIVAL_TS.append(item.enqueued_perf)
        TICK_ARRIVALS += 1
        q_now = len(FIFO)
        TICK_Q_MAX = max(TICK_Q_MAX, q_now)
        RECENT_EVENTS.append(
            {
                "request_id": item.request_id,
                "event": "enqueue",
                "q_sw": q_now,
                "source": item.source,
                "prompt_chars": item.prompt_chars,
            }
        )
    return q_now


def safe_mean(values):
    return round(statistics.mean(values), 2) if values else None


def safe_p95(values):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(0.95 * (len(ordered) - 1))
    return round(ordered[idx], 2)


def build_metrics():
    backend = fetch_backend_metrics()
    power = gpu_snapshot()
    with LOCK:
        q = len(FIFO)
        b = B
        dispatched = DISPATCHED
        completed = COMPLETED
        errors = ERRORS
        tick = TICK
        last_control_source = LAST_CONTROL_SOURCE
        last_control_ts = LAST_CONTROL_TS
        last_tick = dict(LAST_TICK_SUMMARY)
        latencies = list(L_MEAN_BUF)
        ttfts = list(TTFT_BUF)
        qwaits = list(QWAIT_BUF)
        recent_events = list(RECENT_EVENTS)[-10:]
        recent_ticks = list(RECENT_TICKS)[-5:]
        proxy_latencies = list(PROXY_LAT_BUF)
        proxy_ttfts = list(PROXY_TTFT_BUF)
        proxy_errors = PROXY_ERRORS

    metrics = {
        "status": "ok",
        "model": MODEL,
        "backend_url": BACKEND_URL,
        "q_sw": q,
        "B_current": b,
        "B_min": B_MIN,
        "B_max": B_MAX,
        "dt": DT,
        "ticks": tick,
        "dispatched": dispatched,
        "completed": completed,
        "errors": errors,
        "lambda_10s_est": recent_arrival_rate(),
        "l_mean_ms": safe_mean(latencies),
        "l_p95_ms": safe_p95(latencies),
        "ttft_mean_ms": safe_mean(ttfts),
        "ttft_p95_ms": safe_p95(ttfts),
        "queue_wait_mean_ms": safe_mean(qwaits),
        "queue_wait_p95_ms": safe_p95(qwaits),
        "proxy_total_mean_ms": safe_mean(proxy_latencies),
        "proxy_total_p95_ms": safe_p95(proxy_latencies),
        "proxy_ttft_mean_ms": safe_mean(proxy_ttfts),
        "proxy_ttft_p95_ms": safe_p95(proxy_ttfts),
        "proxy_errors": proxy_errors,
        "q_mean_tick": last_tick["q_mean_tick"],
        "q_max_tick": last_tick["q_max_tick"],
        "arrivals_tick": last_tick["arrivals_tick"],
        "completions_tick": last_tick["completions_tick"],
        "service_rate_tick": last_tick["service_rate_tick"],
        "lambda_tick": last_tick["lambda_tick"],
        "vllm_num_requests_waiting": backend.get("vllm:num_requests_waiting"),
        "vllm_num_requests_running": backend.get("vllm:num_requests_running"),
        "vllm_ttft_mean_ms": hist_mean_ms(backend, "vllm:time_to_first_token_seconds"),
        "vllm_e2e_mean_ms": hist_mean_ms(backend, "vllm:e2e_request_latency_seconds"),
        "vllm_queue_mean_ms": hist_mean_ms(backend, "vllm:request_queue_time_seconds"),
        "last_control_source": last_control_source,
        "last_control_ts": last_control_ts,
        "recent_events": recent_events,
        "recent_ticks": recent_ticks,
        "timestamp": datetime.now().isoformat(),
    }
    metrics.update(power)
    metrics.update(scheduler_status())
    return metrics


def prom_metrics_text():
    m = build_metrics()
    lines = []
    gauges = {
        "ch11_q_sw": m["q_sw"],
        "ch11_q_mean_tick": m["q_mean_tick"],
        "ch11_q_max_tick": m["q_max_tick"],
        "ch11_B_current": m["B_current"],
        "ch11_lambda_10s_est": m["lambda_10s_est"],
        "ch11_lambda_tick": m["lambda_tick"],
        "ch11_arrivals_tick": m["arrivals_tick"],
        "ch11_completions_tick": m["completions_tick"],
        "ch11_service_rate_tick": m["service_rate_tick"],
        "ch11_l_mean_ms": m["l_mean_ms"] or 0,
        "ch11_ttft_mean_ms": m["ttft_mean_ms"] or 0,
        "ch11_queue_wait_mean_ms": m["queue_wait_mean_ms"] or 0,
        "ch11_vllm_num_requests_waiting": m["vllm_num_requests_waiting"] or 0,
        "ch11_vllm_num_requests_running": m["vllm_num_requests_running"] or 0,
    }
    for name, value in gauges.items():
        lines.append(f"# TYPE {name} gauge")
        lines.append(f"{name} {value}")
    return "\n".join(lines) + "\n"


def dispatch_one(item, batch_index, result_lock, results):
    global COMPLETED, ERRORS, TICK_COMPLETIONS

    body = {
        "model": MODEL,
        "prompt": item.prompt,
        "max_tokens": item.max_tokens,
        "stream": True,
        "temperature": item.temperature,
    }
    t_dispatch = time.perf_counter()
    q_wait_ms = (t_dispatch - item.enqueued_perf) * 1000.0
    log(
        "dispatch request_id=%s batch_idx=%d q_wait=%.0fms prompt_chars=%d max_tokens=%d"
        % (item.request_id, batch_index, q_wait_ms, item.prompt_chars, item.max_tokens)
    )

    try:
        with requests.post(
            f"{BACKEND_URL}/v1/completions",
            data=json.dumps(body),
            headers=headers(),
            stream=True,
            timeout=TIMEOUT,
        ) as resp:
            resp.raise_for_status()
            first_token = False
            for chunk in resp.iter_lines():
                if chunk and chunk != b"data: [DONE]":
                    t_first = time.perf_counter()
                    ttft_ms = (t_first - t_dispatch) * 1000.0
                    l_total_ms = (t_first - item.enqueued_perf) * 1000.0
                    with LOCK:
                        TTFT_BUF.append(ttft_ms)
                        L_MEAN_BUF.append(l_total_ms)
                        QWAIT_BUF.append(q_wait_ms)
                        COMPLETED += 1
                        TICK_COMPLETIONS += 1
                        RECENT_EVENTS.append(
                            {
                                "request_id": item.request_id,
                                "event": "complete",
                                "ttft_ms": round(ttft_ms, 2),
                                "l_total_ms": round(l_total_ms, 2),
                                "q_wait_ms": round(q_wait_ms, 2),
                            }
                        )
                    with result_lock:
                        results.append((ttft_ms, l_total_ms, q_wait_ms))
                    log(
                        "complete request_id=%s ttft=%.0fms q_wait=%.0fms l_total=%.0fms prompt='%s'"
                        % (
                            item.request_id,
                            ttft_ms,
                            q_wait_ms,
                            l_total_ms,
                            short_prompt(item.prompt),
                        )
                    )
                    first_token = True
                    break
            if not first_token:
                raise RuntimeError("stream ended before first token")
    except Exception as exc:
        with LOCK:
            ERRORS += 1
            RECENT_EVENTS.append(
                {
                    "request_id": item.request_id,
                    "event": "error",
                    "message": str(exc),
                }
            )
        log(f"error request_id={item.request_id} err={exc}")


def dispatcher():
    global TICK, DISPATCHED, TICK_ARRIVALS, TICK_COMPLETIONS, TICK_Q_MAX, LAST_TICK_SUMMARY
    log(
        "dispatcher start backend=%s model=%s dt=%.2fs B=[%d,%d]"
        % (BACKEND_URL, MODEL, DT, B_MIN, B_MAX)
    )
    tick_index = 0
    tick_start = time.perf_counter()
    with LOCK:
        update_queue_area_locked(tick_start)
        area_start = QUEUE_AREA
        TICK_ARRIVALS = 0
        TICK_COMPLETIONS = 0
        TICK_Q_MAX = len(FIFO)
    while True:
        t0 = time.perf_counter()
        tick_index += 1
        with LOCK:
            update_queue_area_locked(t0)
            b_now = B
            batch = []
            while FIFO and len(batch) < b_now:
                batch.append(FIFO.popleft())
            update_queue_area_locked()
            q_after_pop = len(FIFO)
            TICK = tick_index
            tick_now = tick_index
            DISPATCHED += len(batch)

        if batch:
            log(
                "tick=%d dispatch=%d q_after_pop=%d B=%d lambda_10s=%.2f"
                % (tick_now, len(batch), q_after_pop, b_now, recent_arrival_rate())
            )
            result_lock = threading.Lock()
            results = []
            threads = [
                threading.Thread(
                    target=dispatch_one,
                    args=(item, i + 1, result_lock, results),
                    daemon=True,
                )
                for i, item in enumerate(batch)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if results:
                ttfts = [r[0] for r in results]
                lats = [r[1] for r in results]
                waits = [r[2] for r in results]
                backend = fetch_backend_metrics()
                log(
                    "tick=%d summary ttft_mean=%.0fms l_mean=%.0fms q_wait_mean=%.0fms "
                    "vllm_waiting=%s vllm_running=%s"
                    % (
                        tick_now,
                        statistics.mean(ttfts),
                        statistics.mean(lats),
                        statistics.mean(waits),
                        backend.get("vllm:num_requests_waiting"),
                        backend.get("vllm:num_requests_running"),
                    )
                )
        elif tick_now % 5 == 0:
            backend = fetch_backend_metrics()
            log(
                "tick=%d idle q=0 B=%d lambda_10s=%.2f vllm_waiting=%s vllm_running=%s"
                % (
                    tick_now,
                    b_now,
                    recent_arrival_rate(),
                    backend.get("vllm:num_requests_waiting"),
                    backend.get("vllm:num_requests_running"),
                )
            )

        elapsed = time.perf_counter() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)
        tick_end = time.perf_counter()
        with LOCK:
            update_queue_area_locked(tick_end)
            tick_area = QUEUE_AREA - area_start
            tick_elapsed = max(tick_end - tick_start, 1e-6)
            q_mean_tick = tick_area / tick_elapsed
            tick_summary = {
                "tick": tick_now,
                "q_mean_tick": round(q_mean_tick, 2),
                "q_max_tick": int(TICK_Q_MAX),
                "arrivals_tick": int(TICK_ARRIVALS),
                "completions_tick": int(TICK_COMPLETIONS),
                "service_rate_tick": round(TICK_COMPLETIONS / tick_elapsed, 2),
                "lambda_tick": round(TICK_ARRIVALS / tick_elapsed, 2),
            }
            LAST_TICK_SUMMARY = tick_summary
            RECENT_TICKS.append(tick_summary)
            area_start = QUEUE_AREA
            tick_start = tick_end
            TICK_ARRIVALS = 0
            TICK_COMPLETIONS = 0
            TICK_Q_MAX = len(FIFO)
        if batch or tick_summary["arrivals_tick"] > 0 or tick_summary["q_max_tick"] > 0:
            log(
                "tick=%d plant q_mean=%.2f q_max=%d arrivals=%d completions=%d service_rate=%.2f"
                % (
                    tick_now,
                    tick_summary["q_mean_tick"],
                    tick_summary["q_max_tick"],
                    tick_summary["arrivals_tick"],
                    tick_summary["completions_tick"],
                    tick_summary["service_rate_tick"],
                )
            )


def headers():
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["Authorization"] = f"Bearer {API_KEY}"
    return h


def parse_json(handler):
    length = int(handler.headers.get("Content-Length", "0") or "0")
    raw = handler.rfile.read(length).decode("utf-8") if length > 0 else "{}"
    if not raw.strip():
        return {}
    return json.loads(raw)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    server_version = "Chapter8Wrapper/0.1"

    def log_message(self, fmt, *args):
        log("http " + fmt % args)

    def _send_json(self, status, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _send_text(self, status, payload):
        raw = payload.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _proxy_completion(self, body):
        global PROXY_ERRORS
        t_recv = time.perf_counter()
        body = dict(body)
        stream = bool(body.get("stream", False))
        backend_headers = headers()
        try:
            if stream:
                with requests.post(
                    f"{BACKEND_URL}/v1/completions",
                    data=json.dumps(body),
                    headers=backend_headers,
                    stream=True,
                    timeout=TIMEOUT,
                ) as resp:
                    self.send_response(resp.status_code)
                    self.send_header("Content-Type", resp.headers.get("Content-Type", "text/event-stream"))
                    self.end_headers()
                    t_first = None
                    for chunk in resp.iter_lines():
                        if chunk:
                            if t_first is None and chunk != b"data: [DONE]":
                                t_first = time.perf_counter()
                            self.wfile.write(chunk + b"\n\n")
                            self.wfile.flush()
                    t_done = time.perf_counter()
                    if t_first is not None and resp.ok:
                        with LOCK:
                            PROXY_TTFT_BUF.append(1000.0 * (t_first - t_recv))
                            PROXY_LAT_BUF.append(1000.0 * (t_done - t_recv))
                    return

            body["stream"] = False
            resp = requests.post(
                f"{BACKEND_URL}/v1/completions",
                data=json.dumps(body),
                headers=backend_headers,
                timeout=TIMEOUT,
            )
            t_done = time.perf_counter()
            if resp.ok:
                with LOCK:
                    PROXY_LAT_BUF.append(1000.0 * (t_done - t_recv))
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(resp.content)))
            self.end_headers()
            self.wfile.write(resp.content)
        except Exception as exc:
            with LOCK:
                PROXY_ERRORS += 1
            log(f"proxy completion failed: {exc!r}")
            self._send_json(502, {"status": "error", "message": repr(exc)})

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/health":
            health = {"status": "ok", "model": MODEL, "q_sw": len(FIFO), "B": B}
            self._send_json(200, health)
            return
        if parsed.path == "/metrics":
            self._send_json(200, build_metrics())
            return
        if parsed.path == "/metrics/prom":
            self._send_text(200, prom_metrics_text())
            return
        if parsed.path == "/power":
            self._send_json(200, gpu_snapshot())
            return
        self._send_json(404, {"status": "error", "message": "not found"})

    def do_POST(self):
        global B, LAST_CONTROL_SOURCE, LAST_CONTROL_TS

        parsed = urllib.parse.urlparse(self.path)
        try:
            body = parse_json(self)
        except Exception as exc:
            self._send_json(400, {"status": "error", "message": str(exc)})
            return

        if parsed.path == "/v1/completions":
            self._proxy_completion(body)
            return

        if parsed.path == "/control/admission_fraction":
            payload = {
                "mode": "open_loop",
                "admission_fraction": max(0.01, min(1.0, float(body.get("admission_fraction", 1.0)))),
                "enabled": bool(body.get("enabled", True)),
                "source": body.get("source", "http_control"),
                "timestamp": datetime.now().isoformat(),
            }
            with open(CONTROL_FILE, "w") as f:
                json.dump(payload, f)
            log(f"updated scheduler control file {CONTROL_FILE}: {payload}")
            self._send_json(200, {"status": "ok", "control": payload})
            return

        if parsed.path == "/run_internal_budget_sweep":
            result = run_internal_budget_sweep(body)
            self._send_json(200, result)
            return

        if parsed.path == "/control/ttft_target":
            payload = {
                "mode": "ttft",
                "target_ttft_ms": float(body.get("target_ttft_ms", 200.0)),
                "measured_ttft_ms": None,
                "enabled": bool(body.get("enabled", True)),
                "kp": float(body.get("kp", 0.15)),
                "ki": float(body.get("ki", 0.02)),
                "fraction_min": float(body.get("fraction_min", 0.25)),
                "fraction_max": float(body.get("fraction_max", 1.0)),
                "admission_fraction": float(body.get("fraction_max", 1.0)),
                "source": "http_control",
                "timestamp": datetime.now().isoformat(),
            }
            write_scheduler_control(payload)
            log(f"set scheduler to ttft mode {CONTROL_FILE}: {payload}")
            self._send_json(200, {"status": "ok", "control": payload})
            return

        if parsed.path == "/run_internal_ttft_sweep":
            result = run_internal_ttft_sweep(body)
            self._send_json(200, result)
            return

        if parsed.path == "/run_internal_load_step":
            result = run_internal_load_step(body)
            self._send_json(200, result)
            return

        if parsed.path == "/enqueue":
            prompt = body.get("prompt", "")
            prompt_repeat = int(body.get("prompt_repeat", PROMPT_REPEAT_DEFAULT))
            max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
            temperature = float(body.get("temperature", 0.0))
            source = body.get("source", "matlab")
            client_ts = body.get("client_ts", "")
            item = build_queue_item(
                prompt=prompt,
                prompt_repeat=prompt_repeat,
                max_tokens=max_tokens,
                temperature=temperature,
                source=source,
                client_ts=client_ts,
            )
            q_now = enqueue_item(item)
            request_id = item.request_id
            enqueued_wall = item.enqueued_wall
            log(
                "recv enqueue request_id=%s client=%s q=%d prompt_chars=%d repeat=%d max_tokens=%d client_ts=%s prompt='%s'"
                % (
                    request_id,
                    self.client_address[0],
                    q_now,
                    item.prompt_chars,
                    prompt_repeat,
                    max_tokens,
                    client_ts,
                    short_prompt(prompt),
                )
            )
            self._send_json(
                200,
                {
                    "status": "queued",
                    "request_id": request_id,
                    "q_sw": q_now,
                    "timestamp": enqueued_wall,
                },
            )
            return

        if parsed.path == "/enqueue_batch":
            prompt = body.get("prompt", "")
            count = int(body.get("count", 1))
            prompt_repeat = int(body.get("prompt_repeat", PROMPT_REPEAT_DEFAULT))
            max_tokens = int(body.get("max_tokens", MAX_TOKENS_DEFAULT))
            temperature = float(body.get("temperature", 0.0))
            source = body.get("source", "matlab_batch")
            client_ts = body.get("client_ts", "")
            count = max(1, min(count, 1000))

            request_ids = []
            for _ in range(count):
                item = build_queue_item(
                    prompt=prompt,
                    prompt_repeat=prompt_repeat,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    source=source,
                    client_ts=client_ts,
                )
                q_now = enqueue_item(item)
                request_ids.append(item.request_id)

            log(
                "recv enqueue_batch client=%s count=%d q=%d prompt_chars=%d repeat=%d max_tokens=%d source=%s prompt='%s'"
                % (
                    self.client_address[0],
                    count,
                    q_now,
                    len(prompt) * max(prompt_repeat, 1),
                    prompt_repeat,
                    max_tokens,
                    source,
                    short_prompt(prompt),
                )
            )
            self._send_json(
                200,
                {
                    "status": "queued_batch",
                    "count": count,
                    "first_request_id": request_ids[0],
                    "last_request_id": request_ids[-1],
                    "q_sw": q_now,
                    "timestamp": datetime.now().isoformat(),
                },
            )
            return

        if parsed.path == "/control":
            b_new = int(body.get("B", B))
            b_new = max(B_MIN, min(B_MAX, b_new))
            source = body.get("source", "matlab")
            note = body.get("note", "")
            with LOCK:
                B = b_new
                LAST_CONTROL_SOURCE = source
                LAST_CONTROL_TS = datetime.now().isoformat()
                q_now = len(FIFO)
                RECENT_EVENTS.append(
                    {
                        "event": "control",
                        "B": B,
                        "source": source,
                        "q_sw": q_now,
                    }
                )
            log(
                "recv control client=%s set_B=%d q=%d source=%s note='%s'"
                % (self.client_address[0], b_new, q_now, source, note)
            )
            self._send_json(200, {"status": "ok", "B": b_new, "q_sw": q_now})
            return

        if parsed.path == "/reset":
            global DISPATCHED, COMPLETED, ERRORS, TICK
            global QUEUE_AREA, QUEUE_LAST_TS, TICK_ARRIVALS, TICK_COMPLETIONS, TICK_Q_MAX, LAST_TICK_SUMMARY
            with LOCK:
                update_queue_area_locked()
                FIFO.clear()
                update_queue_area_locked()
                L_MEAN_BUF.clear()
                TTFT_BUF.clear()
                QWAIT_BUF.clear()
                ARRIVAL_TS.clear()
                RECENT_EVENTS.clear()
                RECENT_TICKS.clear()
                DISPATCHED = 0
                COMPLETED = 0
                ERRORS = 0
                TICK = 0
                QUEUE_AREA = 0.0
                QUEUE_LAST_TS = time.perf_counter()
                TICK_ARRIVALS = 0
                TICK_COMPLETIONS = 0
                TICK_Q_MAX = 0
                LAST_TICK_SUMMARY = {
                    "tick": 0,
                    "q_mean_tick": 0.0,
                    "q_max_tick": 0,
                    "arrivals_tick": 0,
                    "completions_tick": 0,
                    "service_rate_tick": 0.0,
                    "lambda_tick": 0.0,
                }
            log(f"recv reset client={self.client_address[0]} queue, buffers, and counters cleared")
            self._send_json(200, {"status": "ok"})
            return

        self._send_json(404, {"status": "error", "message": "not found"})


def wait_for_backend():
    log(f"waiting for backend health at {HEALTH_URL}")
    for _ in range(240):
        try:
            resp = requests.get(HEALTH_URL, timeout=5)
            if resp.ok:
                log("backend healthy")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("backend did not become healthy")


def main():
    global TRACE_PREFIX, MODEL, BACKEND_URL, METRICS_URL, HEALTH_URL
    global B, B_MIN, B_MAX, DT, API_KEY, MAX_TOKENS_DEFAULT, PROMPT_REPEAT_DEFAULT, TIMEOUT

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--B-init", type=int, default=4)
    parser.add_argument("--B-min", type=int, default=1)
    parser.add_argument("--B-max", type=int, default=50)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt-repeat", type=int, default=192)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--trace-prefix", default="CH11")
    args = parser.parse_args()

    TRACE_PREFIX = args.trace_prefix
    MODEL = args.model
    BACKEND_URL = args.backend_url
    METRICS_URL = f"{BACKEND_URL}/metrics"
    HEALTH_URL = f"{BACKEND_URL}/health"
    B = args.B_init
    B_MIN = args.B_min
    B_MAX = args.B_max
    DT = args.dt
    API_KEY = args.api_key
    MAX_TOKENS_DEFAULT = args.max_tokens
    PROMPT_REPEAT_DEFAULT = args.prompt_repeat
    TIMEOUT = args.timeout

    wait_for_backend()
    threading.Thread(target=dispatcher, daemon=True).start()
    server = ThreadedHTTPServer((args.host, args.port), Handler)
    log(
        "http server start host=%s port=%d backend=%s B_init=%d B_max=%d dt=%.2f"
        % (args.host, args.port, BACKEND_URL, B, B_MAX, DT)
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
