"""Bayesian Network model definitions."""

from shared.types import BayesianNetworkModel, CPT, Node, Scenario

BATTERY_MODEL = BayesianNetworkModel(
    id="battery",
    name="Level 1: Robot Battery",
    description=(
        "A simple causal chain. Learn how hidden states (Battery) "
        "cause observable effects (Sensors)."
    ),
    nodes=[
        Node(
            id="Battery",
            label="Battery State",
            states=["Full", "Low"],
            node_type="root",
            description="The hidden root cause: is the battery charged or depleted?",
        ),
        Node(
            id="VoltageSensor",
            label="Voltage Sensor",
            states=["High", "Low"],
            node_type="child",
            parents=["Battery"],
            description="Reads battery voltage — noisy measurement.",
        ),
        Node(
            id="WarningLight",
            label="Warning Light",
            states=["Off", "On"],
            node_type="child",
            parents=["Battery"],
            description="Dashboard indicator — triggered by low battery.",
        ),
    ],
    priors={"Battery": {"Full": 0.8, "Low": 0.2}},
    cpts={
        "VoltageSensor": CPT(
            parents=["Battery"],
            table={
                "Full": {"High": 0.9, "Low": 0.1},
                "Low": {"High": 0.2, "Low": 0.8},
            },
        ),
        "WarningLight": CPT(
            parents=["Battery"],
            table={
                "Full": {"Off": 0.95, "On": 0.05},
                "Low": {"Off": 0.1, "On": 0.9},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="Diagnostic: Light On",
            evidence={"WarningLight": "On"},
            description="The warning light is on. How does this update our belief about the battery?",
        ),
        Scenario(
            name="Conflict: Light On + Voltage High",
            evidence={"WarningLight": "On", "VoltageSensor": "High"},
            description="Conflicting signals: the warning light is on but voltage reads high.",
        ),
        Scenario(
            name="d-sep ①  Common Cause: Active",
            evidence={"VoltageSensor": "Low"},
            description=(
                "Structure: VoltageSensor ← Battery → WarningLight (common cause). "
                "Battery is unobserved — the path is ACTIVE. "
                "Even though Voltage and Warning Light have no direct link, observe how "
                "WarningLight shifts: information about one sensor flows through the shared "
                "hidden cause. Formally: VoltageSensor ⊬⊥ WarningLight (marginal dependence)."
            ),
        ),
        Scenario(
            name="d-sep ②  Common Cause: Blocked",
            evidence={"Battery": "Low"},
            description=(
                "Now Battery is observed — the common cause is known. "
                "The path VoltageSensor ← Battery → WarningLight is BLOCKED. "
                "Try toggling VoltageSensor evidence: WarningLight no longer shifts, "
                "because Battery already accounts for everything. "
                "Formally: VoltageSensor ⊥ WarningLight | Battery."
            ),
        ),
    ],
)

SENSOR_FUSION_MODEL = BayesianNetworkModel(
    id="fusion",
    name="Level 2: Sensor Fusion",
    description=(
        "Complex inter-causal reasoning. Weather affects Camera reliability, "
        'enabling "explaining away" of sensor failures.'
    ),
    nodes=[
        Node(
            id="Weather",
            label="Weather Condition",
            states=["Clear", "Fog"],
            node_type="root",
            description="Environmental condition affecting sensor reliability.",
        ),
        Node(
            id="Object",
            label="Object Type",
            states=["Pedestrian", "Obstacle"],
            node_type="root",
            description="The true identity of the object in the car's path.",
        ),
        Node(
            id="Camera",
            label="Camera AI",
            states=["Detected", "None"],
            node_type="child",
            parents=["Weather", "Object"],
            description="Camera-based detection — degraded in fog.",
        ),
        Node(
            id="LiDAR",
            label="LiDAR Sensor",
            states=["Detected", "None"],
            node_type="child",
            parents=["Object"],
            description="LiDAR detection — unaffected by weather.",
        ),
    ],
    priors={
        "Weather": {"Clear": 0.7, "Fog": 0.3},
        "Object": {"Pedestrian": 0.4, "Obstacle": 0.6},
    },
    cpts={
        "Camera": CPT(
            parents=["Weather", "Object"],
            table={
                "Clear,Pedestrian": {"Detected": 0.95, "None": 0.05},
                "Clear,Obstacle": {"Detected": 0.80, "None": 0.20},
                "Fog,Pedestrian": {"Detected": 0.40, "None": 0.60},
                "Fog,Obstacle": {"Detected": 0.20, "None": 0.80},
            },
        ),
        "LiDAR": CPT(
            parents=["Object"],
            table={
                "Pedestrian": {"Detected": 0.90, "None": 0.10},
                "Obstacle": {"Detected": 0.99, "None": 0.01},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="The Foggy Morning",
            evidence={"Weather": "Fog"},
            description="It's foggy. How does this change sensor reliability?",
        ),
        Scenario(
            name="Explaining Away",
            evidence={"Camera": "None", "LiDAR": "Detected", "Weather": "Fog"},
            description=(
                "Camera sees nothing, but LiDAR detects something in fog. "
                "Fog 'explains away' the camera failure."
            ),
        ),
        Scenario(
            name="d-sep ③  V-Structure: Blocked",
            evidence={},
            description=(
                "Structure: Weather → Camera ← Object (v-structure / common effect). "
                "Camera is the collider and is UNOBSERVED — path is BLOCKED. "
                "Weather and Object are root nodes with no direct link and no observed "
                "collider, so they are marginally independent: P(Weather | Object) = P(Weather). "
                "Adjust the Object prior slider — Weather posterior should not move. "
                "Formally: Weather ⊥ Object."
            ),
        ),
        Scenario(
            name="d-sep ④  V-Structure: Active (Collider Observed)",
            evidence={"Camera": "None"},
            description=(
                "Camera (the collider) is now observed. The v-structure path is ACTIVATED. "
                "Weather and Object become conditionally dependent — observing no detection "
                "'explains away' between fog and absence of object. "
                "Adjust the Object prior slider: the Weather posterior now responds. "
                "Formally: Weather ⊬⊥ Object | Camera=None."
            ),
        ),
    ],
)

MISSION_MODEL = BayesianNetworkModel(
    id="mission",
    name="Level 3: Robot Mission Planning",
    description=(
        "A 6-variable network for a mobile robot operating in uncertain terrain. "
        "Demonstrates Variable Elimination, multi-cause interactions, "
        "and diagnostic reasoning from mission outcome back to environment."
    ),
    nodes=[
        Node(
            id="Terrain",
            label="Terrain Difficulty",
            states=["Easy", "Rough"],
            node_type="root",
            description="How challenging is the operating environment? Root cause of sensor and navigation issues.",
        ),
        Node(
            id="Battery",
            label="Battery Level",
            states=["Good", "Low"],
            node_type="root",
            description="Robot's available energy. Low battery reduces navigation reliability.",
        ),
        Node(
            id="Sensors",
            label="Sensor Health",
            states=["Ok", "Degraded"],
            node_type="child",
            parents=["Terrain"],
            description="Rough terrain causes vibrations that degrade IMU and LiDAR sensors.",
        ),
        Node(
            id="Navigation",
            label="Navigation",
            states=["Success", "Failed"],
            node_type="child",
            parents=["Battery"],
            description="Motor power and path planning depend on available energy. Battery is the sole upstream cause of navigation reliability.",
        ),
        Node(
            id="Localization",
            label="Localization",
            states=["Accurate", "Drifted"],
            node_type="child",
            parents=["Sensors", "Terrain"],
            description="Position estimation fails when sensors are degraded or terrain lacks distinguishable features.",
        ),
        Node(
            id="Mission",
            label="Mission Outcome",
            states=["Complete", "Abort"],
            node_type="child",
            parents=["Navigation", "Localization"],
            description="Mission succeeds only if the robot can navigate reliably and knows where it is.",
        ),
    ],
    priors={
        "Terrain": {"Easy": 0.65, "Rough": 0.35},
        "Battery": {"Good": 0.75, "Low": 0.25},
    },
    cpts={
        "Sensors": CPT(
            parents=["Terrain"],
            table={
                "Easy":  {"Ok": 0.95, "Degraded": 0.05},
                "Rough": {"Ok": 0.45, "Degraded": 0.55},
            },
        ),
        "Navigation": CPT(
            parents=["Battery"],
            table={
                "Good": {"Success": 0.88, "Failed": 0.12},
                "Low":  {"Success": 0.40, "Failed": 0.60},
            },
        ),
        "Localization": CPT(
            parents=["Sensors", "Terrain"],
            table={
                "Ok,Easy":       {"Accurate": 0.98, "Drifted": 0.02},
                "Ok,Rough":      {"Accurate": 0.72, "Drifted": 0.28},
                "Degraded,Easy": {"Accurate": 0.55, "Drifted": 0.45},
                "Degraded,Rough":{"Accurate": 0.15, "Drifted": 0.85},
            },
        ),
        "Mission": CPT(
            parents=["Navigation", "Localization"],
            table={
                "Success,Accurate": {"Complete": 0.99, "Abort": 0.01},
                "Success,Drifted":  {"Complete": 0.55, "Abort": 0.45},
                "Failed,Accurate":  {"Complete": 0.35, "Abort": 0.65},
                "Failed,Drifted":   {"Complete": 0.04, "Abort": 0.96},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="Rough Terrain",
            evidence={"Terrain": "Rough"},
            description="The robot enters rough terrain. How does it cascade to sensors, localization, navigation, and mission outcome?",
        ),
        Scenario(
            name="Sensor Failure",
            evidence={"Sensors": "Degraded"},
            description="Diagnostic: sensors report degraded. What does this tell us about terrain, and how does it affect localization and mission?",
        ),
        Scenario(
            name="Full Degradation",
            evidence={"Terrain": "Rough", "Battery": "Low"},
            description="Emergency scenario: rough terrain AND low battery. Observe cascade failure in navigation and mission.",
        ),
        Scenario(
            name="Mission Aborted",
            evidence={"Mission": "Abort"},
            description="The mission was aborted. Work backwards: what does this reveal about the likely environment and robot state?",
        ),
        Scenario(
            name="d-sep ⑤  Causal Chain: Active",
            evidence={"Battery": "Low"},
            description=(
                "Structure: Battery → Navigation → Mission (causal chain). "
                "Navigation is the intermediate node and is UNOBSERVED — path is ACTIVE. "
                "Low battery degrades navigation, which in turn reduces mission success. "
                "Information flows through the chain end-to-end: Battery ⊬⊥ Mission. "
                "(Battery → Navigation is the only path from Battery to Mission, "
                "making this a clean chain with no bypass edges.)"
            ),
        ),
        Scenario(
            name="d-sep ⑥  Causal Chain: Blocked",
            evidence={"Battery": "Low", "Navigation": "Success"},
            description=(
                "Now Navigation (the middle node) is observed = Success. The chain is BLOCKED. "
                "Knowing navigation succeeded, battery level gives no additional information "
                "about mission outcome — the intermediate node screens off the chain. "
                "Compare Mission posteriors with and without Battery evidence while keeping "
                "Navigation fixed: Mission should not shift. "
                "Formally: Battery ⊥ Mission | Navigation."
            ),
        ),
        Scenario(
            name="d-sep ⑦  V-Structure: Active via Effect",
            evidence={"Mission": "Abort"},
            description=(
                "Structure: Navigation → Mission ← Localization (v-structure). "
                "Mission (the collider) is OBSERVED as Abort. The path is ACTIVATED. "
                "Navigation and Localization become conditionally dependent: "
                "knowing the mission failed, 'explaining away' occurs — "
                "if navigation succeeded, localization drift becomes more likely, and vice versa. "
                "Formally: Navigation ⊬⊥ Localization | Mission=Abort."
            ),
        ),
    ],
)

WEATHER_MODEL = BayesianNetworkModel(
    id="weather",
    name="Weather (Sampling Demo)",
    description=(
        "The classic Cloudy → Sprinkler, Cloudy → Rain, Sprinkler+Rain → WetGrass network. "
        "Used in textbooks to illustrate all four BN sampling algorithms. "
        "Try Prior Sampling, Rejection Sampling, Likelihood Weighting, and Gibbs Sampling "
        "and observe how each algorithm converges to the true posterior."
    ),
    nodes=[
        Node(
            id="Cloudy",
            label="Cloudy",
            states=["True", "False"],
            node_type="root",
            description="Is the sky cloudy? Root cause that influences both sprinkler use and rain.",
        ),
        Node(
            id="Sprinkler",
            label="Sprinkler",
            states=["True", "False"],
            node_type="child",
            parents=["Cloudy"],
            description="Is the sprinkler running? Less likely when cloudy (gardeners check the sky).",
        ),
        Node(
            id="Rain",
            label="Rain",
            states=["True", "False"],
            node_type="child",
            parents=["Cloudy"],
            description="Is it raining? More likely when cloudy.",
        ),
        Node(
            id="WetGrass",
            label="Wet Grass",
            states=["True", "False"],
            node_type="child",
            parents=["Sprinkler", "Rain"],
            description="Is the grass wet? Either sprinkler or rain can wet it.",
        ),
    ],
    priors={"Cloudy": {"True": 0.5, "False": 0.5}},
    cpts={
        "Sprinkler": CPT(
            parents=["Cloudy"],
            table={
                "True":  {"True": 0.1, "False": 0.9},
                "False": {"True": 0.5, "False": 0.5},
            },
        ),
        "Rain": CPT(
            parents=["Cloudy"],
            table={
                "True":  {"True": 0.8, "False": 0.2},
                "False": {"True": 0.2, "False": 0.8},
            },
        ),
        "WetGrass": CPT(
            parents=["Sprinkler", "Rain"],
            table={
                "True,True":   {"True": 0.99, "False": 0.01},
                "True,False":  {"True": 0.90, "False": 0.10},
                "False,True":  {"True": 0.90, "False": 0.10},
                "False,False": {"True": 0.01, "False": 0.99},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="Wet Grass Observed",
            evidence={"WetGrass": "True"},
            description=(
                "The grass is wet. What caused it — sprinkler or rain? "
                "This tests diagnostic (bottom-up) reasoning and is ideal for "
                "Likelihood Weighting since WetGrass is a leaf node "
                "(rejection sampling wastes samples when WetGrass=True is unlikely)."
            ),
        ),
        Scenario(
            name="Cloudy + Wet Grass",
            evidence={"Cloudy": "True", "WetGrass": "True"},
            description=(
                "Both an upstream cause and downstream effect are observed. "
                "Compare how Likelihood Weighting and Gibbs Sampling handle mixed "
                "upstream/downstream evidence. Gibbs sampling should converge faster "
                "since it conditions on all evidence simultaneously."
            ),
        ),
        Scenario(
            name="No Rain (rare evidence)",
            evidence={"Rain": "False", "WetGrass": "True"},
            description=(
                "Rare combination: no rain yet the grass is wet — sprinkler must be running. "
                "Rejection sampling will have a very low acceptance rate here. "
                "Observe how Likelihood Weighting and Gibbs Sampling handle this efficiently."
            ),
        ),
    ],
)

# ---------------------------------------------------------------------------
# Model 5: Robot Sensor Fusion (Sampling-focused)
# ---------------------------------------------------------------------------
# Designed to showcase the upstream/downstream evidence asymmetry:
#   - Evidence on roots (Terrain) → all methods work
#   - Evidence on leaves (AccelReading, SpeedSensor) → LW weights collapse,
#     Gibbs converges, rejection wastes samples
#
# Network:
#   Terrain ──→ Vibration ──→ AccelReading
#   Terrain ──→ WheelSlip ──→ SpeedSensor
#   BatteryAge ──→ WheelSlip
#
# BatteryAge and Terrain are co-parents of WheelSlip → explaining away.
# AccelReading and SpeedSensor are leaf nodes → ideal for demonstrating
# why Gibbs beats LW when evidence is downstream.
# ---------------------------------------------------------------------------

ROBOT_SAMPLING_MODEL = BayesianNetworkModel(
    id="robot_sampling",
    name="Robot Diagnostics (Gibbs Demo)",
    description=(
        "A 6-node robot diagnostics network with leaf-node sensor readings. "
        "Demonstrates why Gibbs Sampling outperforms Likelihood Weighting "
        "when evidence is at the leaves (downstream), and showcases "
        "explaining away between Terrain and BatteryAge."
    ),
    nodes=[
        Node(
            id="Terrain",
            label="Terrain",
            states=["Easy", "Rough"],
            node_type="root",
            description="Operating environment. Rough terrain causes vibrations and wheel slip.",
        ),
        Node(
            id="BatteryAge",
            label="Battery Age",
            states=["New", "Old"],
            node_type="root",
            description="Age of the battery pack. Old batteries cause degraded traction control → more wheel slip.",
        ),
        Node(
            id="Vibration",
            label="Vibration Level",
            states=["Low", "High"],
            node_type="child",
            parents=["Terrain"],
            description="Chassis vibration measured by IMU. Rough terrain causes high vibration.",
        ),
        Node(
            id="WheelSlip",
            label="Wheel Slip",
            states=["None", "Slip"],
            node_type="child",
            parents=["Terrain", "BatteryAge"],
            description=(
                "Traction loss. Both rough terrain and old batteries increase slip risk. "
                "This is a collider — Terrain and BatteryAge become dependent when WheelSlip is observed."
            ),
        ),
        Node(
            id="AccelReading",
            label="Accelerometer",
            states=["Normal", "Alert"],
            node_type="child",
            parents=["Vibration"],
            description="Accelerometer sensor reading. High vibration triggers alert, but the sensor has noise.",
        ),
        Node(
            id="SpeedSensor",
            label="Speed Sensor",
            states=["Accurate", "Drifted"],
            node_type="child",
            parents=["WheelSlip"],
            description="Wheel-speed encoder. Slip causes erratic readings (drift).",
        ),
    ],
    priors={
        "Terrain": {"Easy": 0.6, "Rough": 0.4},
        "BatteryAge": {"New": 0.7, "Old": 0.3},
    },
    cpts={
        "Vibration": CPT(
            parents=["Terrain"],
            table={
                "Easy":  {"Low": 0.90, "High": 0.10},
                "Rough": {"Low": 0.25, "High": 0.75},
            },
        ),
        "WheelSlip": CPT(
            parents=["Terrain", "BatteryAge"],
            table={
                "Easy,New":   {"None": 0.95, "Slip": 0.05},
                "Easy,Old":   {"None": 0.75, "Slip": 0.25},
                "Rough,New":  {"None": 0.55, "Slip": 0.45},
                "Rough,Old":  {"None": 0.20, "Slip": 0.80},
            },
        ),
        "AccelReading": CPT(
            parents=["Vibration"],
            table={
                "Low":  {"Normal": 0.92, "Alert": 0.08},
                "High": {"Normal": 0.15, "Alert": 0.85},
            },
        ),
        "SpeedSensor": CPT(
            parents=["WheelSlip"],
            table={
                "None": {"Accurate": 0.95, "Drifted": 0.05},
                "Slip": {"Accurate": 0.20, "Drifted": 0.80},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="① Upstream evidence (easy for LW)",
            evidence={"Terrain": "Rough"},
            description=(
                "Evidence on a root node. Likelihood Weighting works well here "
                "because downstream variables are sampled conditioned on the "
                "observed root. Compare all four methods — they should all converge "
                "quickly."
            ),
        ),
        Scenario(
            name="② Leaf evidence (LW struggles)",
            evidence={"AccelReading": "Alert", "SpeedSensor": "Drifted"},
            description=(
                "Both sensors (leaf nodes) report problems. What caused it? "
                "Likelihood Weighting samples Terrain and BatteryAge from the prior — "
                "unaware of the leaf evidence — so weights collapse. "
                "Gibbs Sampling conditions each variable on all evidence via the "
                "Markov blanket, converging much faster. "
                "Watch the convergence chart: Gibbs (green) converges while LW (orange) stays noisy."
            ),
        ),
        Scenario(
            name="③ Explaining away",
            evidence={"WheelSlip": "Slip", "BatteryAge": "Old"},
            description=(
                "We observe slip and know the battery is old. "
                "The old battery already 'explains' the slip, so P(Terrain=Rough) drops "
                "compared to observing slip alone. This is explaining away: "
                "one cause reduces the need for the other. "
                "Try removing BatteryAge evidence to see Terrain probability rise."
            ),
        ),
        Scenario(
            name="④ Rare leaf evidence (rejection fails)",
            evidence={"AccelReading": "Alert", "SpeedSensor": "Drifted", "Vibration": "Low"},
            description=(
                "Contradictory evidence: accelerometer alerts but vibration is low, "
                "plus the speed sensor drifts. This combination is rare (~2% of prior samples). "
                "Rejection sampling acceptance rate will be very low. "
                "Likelihood Weighting produces tiny weights. "
                "Gibbs Sampling handles this gracefully."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Model 6: Student Grades
# ---------------------------------------------------------------------------
# Lightweight model for hand-tracing Gibbs steps on paper.
# Based on the classic Koller & Friedman "Student" network, adapted.
#
# Network:
#   Difficulty ──→ Grade ←── HardWorking
#   HardWorking ──→ SAT
#   Grade ──→ Letter
#
# Explaining away: Letter=Strong. Was the course easy or the student hard-working?
# Adding SAT=High resolves the ambiguity.
# ---------------------------------------------------------------------------

STUDENT_MODEL = BayesianNetworkModel(
    id="student",
    name="Student Grades",
    description=(
        "A 5-node model inspired by Koller & Friedman. "
        "A student's Grade depends on course Difficulty and how Hard-working they are. "
        "The Grade determines the recommendation Letter, and Hard-working also "
        "affects the SAT score. Ideal for hand-tracing Gibbs steps and "
        "observing explaining away between Difficulty and Hard-working."
    ),
    nodes=[
        Node(
            id="Difficulty",
            label="Course Difficulty",
            states=["Easy", "Hard"],
            node_type="root",
            description="How hard is the course? Easy courses yield high grades more often.",
        ),
        Node(
            id="HardWorking",
            label="Hard-working",
            states=["Low", "High"],
            node_type="root",
            description="Student's work ethic. Affects both grade and SAT performance.",
        ),
        Node(
            id="Grade",
            label="Grade",
            states=["A", "B", "C"],
            node_type="child",
            parents=["Difficulty", "HardWorking"],
            description=(
                "Course grade. Depends on both difficulty and work ethic. "
                "This is a collider — observing Grade activates the path "
                "between Difficulty and Hard-working (explaining away)."
            ),
        ),
        Node(
            id="SAT",
            label="SAT Score",
            states=["Low", "High"],
            node_type="child",
            parents=["HardWorking"],
            description="Standardised test score. Directly reflects work ethic, independent of courses.",
        ),
        Node(
            id="Letter",
            label="Rec. Letter",
            states=["Weak", "Strong"],
            node_type="child",
            parents=["Grade"],
            description="Strength of the professor's recommendation letter. Directly based on the grade.",
        ),
    ],
    priors={
        "Difficulty": {"Easy": 0.6, "Hard": 0.4},
        "HardWorking": {"Low": 0.7, "High": 0.3},
    },
    cpts={
        "Grade": CPT(
            parents=["Difficulty", "HardWorking"],
            table={
                "Easy,Low":  {"A": 0.30, "B": 0.40, "C": 0.30},
                "Easy,High": {"A": 0.90, "B": 0.08, "C": 0.02},
                "Hard,Low":  {"A": 0.05, "B": 0.25, "C": 0.70},
                "Hard,High": {"A": 0.50, "B": 0.30, "C": 0.20},
            },
        ),
        "SAT": CPT(
            parents=["HardWorking"],
            table={
                "Low":  {"Low": 0.95, "High": 0.05},
                "High": {"Low": 0.20, "High": 0.80},
            },
        ),
        "Letter": CPT(
            parents=["Grade"],
            table={
                "A": {"Weak": 0.05, "Strong": 0.95},
                "B": {"Weak": 0.40, "Strong": 0.60},
                "C": {"Weak": 0.90, "Strong": 0.10},
            },
        ),
    },
    scenarios=[
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="① Strong letter (leaf evidence)",
            evidence={"Letter": "Strong"},
            description=(
                "The student got a strong recommendation. Was the course easy "
                "or is the student hard-working? Both could explain a good grade. "
                "This is leaf evidence — Gibbs propagates it back through Grade "
                "to both root causes, while LW samples Difficulty and HardWorking "
                "from the prior (ignoring the letter entirely in the sampling step)."
            ),
        ),
        Scenario(
            name="② Strong letter + High SAT",
            evidence={"Letter": "Strong", "SAT": "High"},
            description=(
                "Adding SAT=High resolves the ambiguity: the student is hard-working. "
                "P(HardWorking=High) jumps, and P(Difficulty=Hard) can now increase too "
                "(a diligent student can get a good grade even in a hard course). "
                "Compare convergence speed: Gibbs mixes faster with more evidence."
            ),
        ),
        Scenario(
            name="③ Explaining away: Grade observed",
            evidence={"Grade": "A"},
            description=(
                "Grade is the collider between Difficulty and Hard-working. "
                "Observing Grade=A activates the path: now Difficulty and Hard-working "
                "are dependent. If we also learn the course is Easy, the need for "
                "high effort diminishes — explaining away. "
                "Try adding Difficulty=Easy evidence and watch HardWorking shift."
            ),
        ),
        Scenario(
            name="④ Diagnostic: weak letter, hard course",
            evidence={"Letter": "Weak", "Difficulty": "Hard"},
            description=(
                "Mixed evidence: upstream (Difficulty=Hard) and downstream (Letter=Weak). "
                "The hard course makes a bad grade more likely, which explains the weak letter. "
                "But is the student also lazy? Check P(HardWorking) — "
                "the hard course partially 'explains' the poor outcome, softening the "
                "inference about work ethic."
            ),
        ),
    ],
)


# ---------------------------------------------------------------------------
# Model 7: Ising Grid (Image Denoising)
# ---------------------------------------------------------------------------
# A 4x4 binary grid (16 nodes) where each pixel depends on its spatial
# neighbours. Observed: noisy version of a simple pattern.
#
# This is a Markov Random Field rewritten as a BN by imposing a
# topological order (raster scan: left-to-right, top-to-bottom).
# Each pixel conditions on its already-visited neighbours (up and left).
#
# Demonstrates:
#   - Gibbs at scale (16 variables, rejection acceptance rate ~ 2^{-16})
#   - Spatial locality of the Markov blanket (4 neighbours)
#   - Image "crystallisation" during burn-in
# ---------------------------------------------------------------------------

def _make_ising_model() -> BayesianNetworkModel:
    """Build a 4x4 Ising-like grid model for image denoising.

    Topology (raster-scan BN encoding of an MRF):
    Each pixel P_rc (row r, col c) has parents = {P above, P left} when they
    exist.  The CPT encodes spatial smoothness: neighbouring pixels prefer to
    agree (coupling strength ~0.85).

    Noisy observations: each pixel has a child O_rc (observed pixel) that
    flips with noise probability 0.15.
    """
    rows, cols = 4, 4
    coupling = 0.85  # P(same as neighbour)
    noise = 0.15     # P(observed pixel != true pixel)
    nodes: list[Node] = []
    priors: dict[str, dict[str, float]] = {}
    cpts: dict[str, CPT] = {}

    # --- hidden pixel nodes (raster-scan BN) ---
    for r in range(rows):
        for c in range(cols):
            pid = f"P{r}{c}"
            parents: list[str] = []
            if r > 0:
                parents.append(f"P{r-1}{c}")  # pixel above
            if c > 0:
                parents.append(f"P{r}{c-1}")  # pixel left

            nodes.append(
                Node(
                    id=pid,
                    label=f"Pixel ({r},{c})",
                    states=["Black", "White"],
                    node_type="root" if not parents else "child",
                    parents=parents,
                    description=f"Hidden true pixel at row {r}, col {c}.",
                )
            )

            if not parents:
                # top-left corner: uniform prior
                priors[pid] = {"Black": 0.5, "White": 0.5}
            else:
                # CPT: prefer to agree with each parent
                table: dict[str, dict[str, float]] = {}
                parent_states_list = [["Black", "White"]] * len(parents)
                # build cartesian product of parent states
                import itertools
                for combo in itertools.product(*parent_states_list):
                    key = ",".join(combo)
                    n_black = combo.count("Black")
                    n_white = combo.count("White")
                    # bias toward majority of parents
                    if n_black > n_white:
                        p_black = coupling
                    elif n_white > n_black:
                        p_black = 1 - coupling
                    else:
                        p_black = 0.5
                    table[key] = {"Black": round(p_black, 4), "White": round(1 - p_black, 4)}
                cpts[pid] = CPT(parents=parents, table=table)

            # --- noisy observation node ---
            oid = f"O{r}{c}"
            nodes.append(
                Node(
                    id=oid,
                    label=f"Obs ({r},{c})",
                    states=["Black", "White"],
                    node_type="child",
                    parents=[pid],
                    description=f"Noisy observation of pixel ({r},{c}). Flips with probability {noise}.",
                )
            )
            cpts[oid] = CPT(
                parents=[pid],
                table={
                    "Black": {"Black": round(1 - noise, 2), "White": round(noise, 2)},
                    "White": {"Black": round(noise, 2), "White": round(1 - noise, 2)},
                },
            )

    # --- scenarios: observe a noisy cross pattern ---
    # True pattern: cross shape (rows 1-2 col 1-2 center area)
    # Black = background, White = cross
    true_pattern = [
        ["Black", "White", "White", "Black"],
        ["White", "White", "White", "White"],
        ["White", "White", "White", "White"],
        ["Black", "White", "White", "Black"],
    ]
    # Noisy version: flip a few pixels
    noisy_cross: dict[str, str] = {}
    flip_positions = {(0, 0), (1, 3), (2, 0), (3, 3)}  # pixels that got flipped by noise
    for r in range(rows):
        for c in range(cols):
            true_val = true_pattern[r][c]
            if (r, c) in flip_positions:
                noisy_cross[f"O{r}{c}"] = "White" if true_val == "Black" else "Black"
            else:
                noisy_cross[f"O{r}{c}"] = true_val

    # Clean observation (no noise) for comparison
    clean_obs: dict[str, str] = {}
    for r in range(rows):
        for c in range(cols):
            clean_obs[f"O{r}{c}"] = true_pattern[r][c]

    # Partial observation: only observe edges
    partial_obs: dict[str, str] = {}
    for r in range(rows):
        for c in range(cols):
            if r == 0 or r == 3 or c == 0 or c == 3:
                partial_obs[f"O{r}{c}"] = true_pattern[r][c]

    scenarios = [
        Scenario(name="Reset (Prior)", evidence={}),
        Scenario(
            name="① Noisy cross (denoise it)",
            evidence=noisy_cross,
            description=(
                "A cross pattern was photographed with a noisy sensor (15% flip rate). "
                "4 pixels are corrupted. Can Gibbs Sampling recover the true image? "
                "Rejection sampling is hopeless: acceptance rate ~ 2^{-16}. "
                "LW weights collapse because all 16 observations are leaves. "
                "Gibbs resamples each hidden pixel conditioned on its 4 spatial neighbours "
                "— exactly the Markov blanket — and the cross crystallises within ~200 steps."
            ),
        ),
        Scenario(
            name="② Clean observation",
            evidence=clean_obs,
            description=(
                "All 16 pixels observed without noise. The posterior over hidden pixels "
                "should be nearly certain. Even Gibbs converges instantly — a sanity check."
            ),
        ),
        Scenario(
            name="③ Partial observation (edges only)",
            evidence=partial_obs,
            description=(
                "Only the 12 border pixels are observed (the 4 interior pixels are hidden). "
                "The spatial coupling prior propagates border information inward. "
                "Watch how Gibbs infers the interior from boundary conditions — "
                "analogous to how a Gaussian MRF fills in missing values in GP regression."
            ),
        ),
    ]

    return BayesianNetworkModel(
        id="ising",
        name="Image Denoising (Ising Grid)",
        description=(
            "A 4×4 pixel grid where each pixel depends on its spatial neighbours. "
            "Noisy observations of a cross pattern are given; the goal is to recover "
            "the true image. This is the canonical use case for Gibbs Sampling: "
            "exact inference is intractable, LW weights collapse on 16 leaf nodes, "
            "but Gibbs converges by exploiting the local Markov blanket structure."
        ),
        nodes=nodes,
        priors=priors,
        cpts=cpts,
        scenarios=scenarios,
    )


ISING_MODEL = _make_ising_model()


# --- Medical Diagnosis Network (15 nodes) ---
# Demonstrates why different sampling methods converge at different rates
MEDICAL_DIAGNOSIS_MODEL = BayesianNetworkModel(
    id="medical_diagnosis",
    name="Medical Diagnosis (15 nodes)",
    description=(
        "A realistic medical diagnostic network where symptoms and test results "
        "must be used to infer underlying diseases. With evidence (symptoms/tests), "
        "rejection sampling fails (low acceptance), but Gibbs Sampling excels. "
        "Perfect for comparing sampling methods side-by-side."
    ),
    nodes=[
        # Risk factors (roots)
        Node(id="Age", label="Age Group", states=["Young", "Middle", "Old"], node_type="root",
             description="Patient age: affects disease susceptibility."),
        Node(id="Smoking", label="Smoking Status", states=["Yes", "No"], node_type="root",
             description="Current or past smoking: risk factor for respiratory diseases."),
        Node(id="Vaccination", label="Vaccination Status", states=["Vaccinated", "Unvaccinated"], node_type="root",
             description="COVID/Flu vaccination: reduces disease probability."),
        # Diseases (depend on risk factors)
        Node(id="Flu", label="Influenza", states=["Yes", "No"], node_type="child", parents=["Age", "Smoking", "Vaccination"],
             description="Patient has active flu infection."),
        Node(id="COVID", label="COVID-19", states=["Yes", "No"], node_type="child", parents=["Age", "Vaccination"],
             description="Patient has active COVID-19 infection."),
        Node(id="Pneumonia", label="Pneumonia", states=["Yes", "No"], node_type="child", parents=["Age", "Smoking"],
             description="Patient has bacterial or viral pneumonia."),
        # Symptoms (depend on diseases)
        Node(id="Fever", label="Fever", states=["Yes", "No"], node_type="child", parents=["Flu", "COVID", "Pneumonia"],
             description="Elevated body temperature."),
        Node(id="Cough", label="Cough", states=["Yes", "No"], node_type="child", parents=["Flu", "COVID", "Pneumonia"],
             description="Persistent cough."),
        Node(id="Fatigue", label="Fatigue", states=["Yes", "No"], node_type="child", parents=["Flu", "COVID", "Pneumonia"],
             description="Extreme tiredness."),
        Node(id="SOB", label="Shortness of Breath", states=["Yes", "No"], node_type="child", parents=["COVID", "Pneumonia"],
             description="Difficulty breathing."),
        Node(id="SoreThroat", label="Sore Throat", states=["Yes", "No"], node_type="child", parents=["Flu", "COVID"],
             description="Throat pain or irritation."),
        # Lab tests (depend on diseases)
        Node(id="PCRTest", label="PCR Test Result", states=["Positive", "Negative"], node_type="child", parents=["COVID"],
             description="Highly specific COVID-19 test."),
        Node(id="AntigenTest", label="Antigen Test Result", states=["Positive", "Negative"], node_type="child", parents=["Flu", "COVID"],
             description="Rapid test for Flu or COVID."),
        Node(id="ChestXray", label="Chest X-ray", states=["Clear", "Infiltrates"], node_type="child", parents=["Pneumonia", "COVID"],
             description="Lung imaging: detects infection patterns."),
        # Outcome (depends on diseases and severity factors)
        Node(id="Severity", label="Clinical Severity", states=["Mild", "Moderate", "Severe"], node_type="child",
             parents=["Flu", "COVID", "Pneumonia", "Age", "SOB"],
             description="Overall clinical severity assessment."),
    ],
    priors={
        "Age": {"Young": 0.3, "Middle": 0.5, "Old": 0.2},
        "Smoking": {"Yes": 0.2, "No": 0.8},
        "Vaccination": {"Vaccinated": 0.6, "Unvaccinated": 0.4},
    },
    cpts={
        "Flu": CPT(
            parents=["Age", "Smoking", "Vaccination"],
            table={
                "Young,Yes,Vaccinated": {"Yes": 0.05, "No": 0.95},
                "Young,Yes,Unvaccinated": {"Yes": 0.15, "No": 0.85},
                "Young,No,Vaccinated": {"Yes": 0.03, "No": 0.97},
                "Young,No,Unvaccinated": {"Yes": 0.10, "No": 0.90},
                "Middle,Yes,Vaccinated": {"Yes": 0.06, "No": 0.94},
                "Middle,Yes,Unvaccinated": {"Yes": 0.18, "No": 0.82},
                "Middle,No,Vaccinated": {"Yes": 0.04, "No": 0.96},
                "Middle,No,Unvaccinated": {"Yes": 0.12, "No": 0.88},
                "Old,Yes,Vaccinated": {"Yes": 0.08, "No": 0.92},
                "Old,Yes,Unvaccinated": {"Yes": 0.25, "No": 0.75},
                "Old,No,Vaccinated": {"Yes": 0.06, "No": 0.94},
                "Old,No,Unvaccinated": {"Yes": 0.20, "No": 0.80},
            },
        ),
        "COVID": CPT(
            parents=["Age", "Vaccination"],
            table={
                "Young,Vaccinated": {"Yes": 0.02, "No": 0.98},
                "Young,Unvaccinated": {"Yes": 0.08, "No": 0.92},
                "Middle,Vaccinated": {"Yes": 0.03, "No": 0.97},
                "Middle,Unvaccinated": {"Yes": 0.12, "No": 0.88},
                "Old,Vaccinated": {"Yes": 0.05, "No": 0.95},
                "Old,Unvaccinated": {"Yes": 0.20, "No": 0.80},
            },
        ),
        "Pneumonia": CPT(
            parents=["Age", "Smoking"],
            table={
                "Young,Yes": {"Yes": 0.04, "No": 0.96},
                "Young,No": {"Yes": 0.01, "No": 0.99},
                "Middle,Yes": {"Yes": 0.06, "No": 0.94},
                "Middle,No": {"Yes": 0.02, "No": 0.98},
                "Old,Yes": {"Yes": 0.12, "No": 0.88},
                "Old,No": {"Yes": 0.05, "No": 0.95},
            },
        ),
        "Fever": CPT(
            parents=["Flu", "COVID", "Pneumonia"],
            table={
                "Yes,Yes,Yes": {"Yes": 0.95, "No": 0.05},
                "Yes,Yes,No": {"Yes": 0.90, "No": 0.10},
                "Yes,No,Yes": {"Yes": 0.85, "No": 0.15},
                "Yes,No,No": {"Yes": 0.70, "No": 0.30},
                "No,Yes,Yes": {"Yes": 0.80, "No": 0.20},
                "No,Yes,No": {"Yes": 0.75, "No": 0.25},
                "No,No,Yes": {"Yes": 0.70, "No": 0.30},
                "No,No,No": {"Yes": 0.05, "No": 0.95},
            },
        ),
        "Cough": CPT(
            parents=["Flu", "COVID", "Pneumonia"],
            table={
                "Yes,Yes,Yes": {"Yes": 0.95, "No": 0.05},
                "Yes,Yes,No": {"Yes": 0.88, "No": 0.12},
                "Yes,No,Yes": {"Yes": 0.90, "No": 0.10},
                "Yes,No,No": {"Yes": 0.80, "No": 0.20},
                "No,Yes,Yes": {"Yes": 0.85, "No": 0.15},
                "No,Yes,No": {"Yes": 0.70, "No": 0.30},
                "No,No,Yes": {"Yes": 0.85, "No": 0.15},
                "No,No,No": {"Yes": 0.10, "No": 0.90},
            },
        ),
        "Fatigue": CPT(
            parents=["Flu", "COVID", "Pneumonia"],
            table={
                "Yes,Yes,Yes": {"Yes": 0.90, "No": 0.10},
                "Yes,Yes,No": {"Yes": 0.85, "No": 0.15},
                "Yes,No,Yes": {"Yes": 0.75, "No": 0.25},
                "Yes,No,No": {"Yes": 0.65, "No": 0.35},
                "No,Yes,Yes": {"Yes": 0.80, "No": 0.20},
                "No,Yes,No": {"Yes": 0.70, "No": 0.30},
                "No,No,Yes": {"Yes": 0.60, "No": 0.40},
                "No,No,No": {"Yes": 0.10, "No": 0.90},
            },
        ),
        "SOB": CPT(
            parents=["COVID", "Pneumonia"],
            table={
                "Yes,Yes": {"Yes": 0.85, "No": 0.15},
                "Yes,No": {"Yes": 0.40, "No": 0.60},
                "No,Yes": {"Yes": 0.70, "No": 0.30},
                "No,No": {"Yes": 0.05, "No": 0.95},
            },
        ),
        "SoreThroat": CPT(
            parents=["Flu", "COVID"],
            table={
                "Yes,Yes": {"Yes": 0.80, "No": 0.20},
                "Yes,No": {"Yes": 0.75, "No": 0.25},
                "No,Yes": {"Yes": 0.60, "No": 0.40},
                "No,No": {"Yes": 0.05, "No": 0.95},
            },
        ),
        "PCRTest": CPT(
            parents=["COVID"],
            table={
                "Yes": {"Positive": 0.95, "Negative": 0.05},  # 95% sensitivity
                "No": {"Positive": 0.01, "Negative": 0.99},   # 1% false positive
            },
        ),
        "AntigenTest": CPT(
            parents=["Flu", "COVID"],
            table={
                "Yes,Yes": {"Positive": 0.88, "Negative": 0.12},
                "Yes,No": {"Positive": 0.80, "Negative": 0.20},
                "No,Yes": {"Positive": 0.75, "Negative": 0.25},
                "No,No": {"Positive": 0.02, "Negative": 0.98},
            },
        ),
        "ChestXray": CPT(
            parents=["Pneumonia", "COVID"],
            table={
                "Yes,Yes": {"Clear": 0.10, "Infiltrates": 0.90},
                "Yes,No": {"Clear": 0.15, "Infiltrates": 0.85},
                "No,Yes": {"Clear": 0.30, "Infiltrates": 0.70},
                "No,No": {"Clear": 0.95, "Infiltrates": 0.05},
            },
        ),
        "Severity": CPT(
            parents=["Flu", "COVID", "Pneumonia", "Age", "SOB"],
            table={
                "Yes,Yes,Yes,Young,Yes": {"Mild": 0.05, "Moderate": 0.25, "Severe": 0.70},
                "Yes,Yes,Yes,Young,No": {"Mild": 0.10, "Moderate": 0.50, "Severe": 0.40},
                "Yes,Yes,Yes,Middle,Yes": {"Mild": 0.02, "Moderate": 0.20, "Severe": 0.78},
                "Yes,Yes,Yes,Middle,No": {"Mild": 0.08, "Moderate": 0.40, "Severe": 0.52},
                "Yes,Yes,Yes,Old,Yes": {"Mild": 0.01, "Moderate": 0.15, "Severe": 0.84},
                "Yes,Yes,Yes,Old,No": {"Mild": 0.05, "Moderate": 0.30, "Severe": 0.65},
                "Yes,Yes,No,Young,Yes": {"Mild": 0.10, "Moderate": 0.40, "Severe": 0.50},
                "Yes,Yes,No,Young,No": {"Mild": 0.20, "Moderate": 0.60, "Severe": 0.20},
                "Yes,Yes,No,Middle,Yes": {"Mild": 0.08, "Moderate": 0.35, "Severe": 0.57},
                "Yes,Yes,No,Middle,No": {"Mild": 0.18, "Moderate": 0.55, "Severe": 0.27},
                "Yes,Yes,No,Old,Yes": {"Mild": 0.05, "Moderate": 0.30, "Severe": 0.65},
                "Yes,Yes,No,Old,No": {"Mild": 0.12, "Moderate": 0.45, "Severe": 0.43},
                "Yes,No,Yes,Young,Yes": {"Mild": 0.15, "Moderate": 0.50, "Severe": 0.35},
                "Yes,No,Yes,Young,No": {"Mild": 0.25, "Moderate": 0.60, "Severe": 0.15},
                "Yes,No,Yes,Middle,Yes": {"Mild": 0.12, "Moderate": 0.45, "Severe": 0.43},
                "Yes,No,Yes,Middle,No": {"Mild": 0.22, "Moderate": 0.58, "Severe": 0.20},
                "Yes,No,Yes,Old,Yes": {"Mild": 0.08, "Moderate": 0.40, "Severe": 0.52},
                "Yes,No,Yes,Old,No": {"Mild": 0.15, "Moderate": 0.50, "Severe": 0.35},
                "Yes,No,No,Young,Yes": {"Mild": 0.30, "Moderate": 0.55, "Severe": 0.15},
                "Yes,No,No,Young,No": {"Mild": 0.45, "Moderate": 0.45, "Severe": 0.10},
                "Yes,No,No,Middle,Yes": {"Mild": 0.25, "Moderate": 0.50, "Severe": 0.25},
                "Yes,No,No,Middle,No": {"Mild": 0.40, "Moderate": 0.50, "Severe": 0.10},
                "Yes,No,No,Old,Yes": {"Mild": 0.15, "Moderate": 0.45, "Severe": 0.40},
                "Yes,No,No,Old,No": {"Mild": 0.30, "Moderate": 0.50, "Severe": 0.20},
                "No,Yes,Yes,Young,Yes": {"Mild": 0.08, "Moderate": 0.35, "Severe": 0.57},
                "No,Yes,Yes,Young,No": {"Mild": 0.15, "Moderate": 0.50, "Severe": 0.35},
                "No,Yes,Yes,Middle,Yes": {"Mild": 0.05, "Moderate": 0.30, "Severe": 0.65},
                "No,Yes,Yes,Middle,No": {"Mild": 0.12, "Moderate": 0.45, "Severe": 0.43},
                "No,Yes,Yes,Old,Yes": {"Mild": 0.02, "Moderate": 0.20, "Severe": 0.78},
                "No,Yes,Yes,Old,No": {"Mild": 0.08, "Moderate": 0.35, "Severe": 0.57},
                "No,Yes,No,Young,Yes": {"Mild": 0.25, "Moderate": 0.55, "Severe": 0.20},
                "No,Yes,No,Young,No": {"Mild": 0.40, "Moderate": 0.50, "Severe": 0.10},
                "No,Yes,No,Middle,Yes": {"Mild": 0.20, "Moderate": 0.50, "Severe": 0.30},
                "No,Yes,No,Middle,No": {"Mild": 0.35, "Moderate": 0.50, "Severe": 0.15},
                "No,Yes,No,Old,Yes": {"Mild": 0.10, "Moderate": 0.40, "Severe": 0.50},
                "No,Yes,No,Old,No": {"Mild": 0.20, "Moderate": 0.50, "Severe": 0.30},
                "No,No,Yes,Young,Yes": {"Mild": 0.20, "Moderate": 0.55, "Severe": 0.25},
                "No,No,Yes,Young,No": {"Mild": 0.30, "Moderate": 0.60, "Severe": 0.10},
                "No,No,Yes,Middle,Yes": {"Mild": 0.15, "Moderate": 0.50, "Severe": 0.35},
                "No,No,Yes,Middle,No": {"Mild": 0.25, "Moderate": 0.60, "Severe": 0.15},
                "No,No,Yes,Old,Yes": {"Mild": 0.08, "Moderate": 0.40, "Severe": 0.52},
                "No,No,Yes,Old,No": {"Mild": 0.15, "Moderate": 0.50, "Severe": 0.35},
                "No,No,No,Young,Yes": {"Mild": 0.50, "Moderate": 0.45, "Severe": 0.05},
                "No,No,No,Young,No": {"Mild": 0.70, "Moderate": 0.25, "Severe": 0.05},
                "No,No,No,Middle,Yes": {"Mild": 0.45, "Moderate": 0.48, "Severe": 0.07},
                "No,No,No,Middle,No": {"Mild": 0.65, "Moderate": 0.30, "Severe": 0.05},
                "No,No,No,Old,Yes": {"Mild": 0.35, "Moderate": 0.55, "Severe": 0.10},
                "No,No,No,Old,No": {"Mild": 0.50, "Moderate": 0.45, "Severe": 0.05},
            },
        ),
    },
    scenarios=[
        Scenario(
            name="Reset (Prior)",
            evidence={},
            description="No observations. See prior probabilities of diseases.",
        ),
        Scenario(
            name="① Flu-like symptoms (no test)",
            evidence={"Fever": "Yes", "Cough": "Yes", "SoreThroat": "Yes", "Fatigue": "Yes"},
            description=(
                "Patient reports classic flu symptoms. Rejection sampling has moderate acceptance "
                "(~1-2%). Gibbs will quickly converge to high P(Flu), moderate P(COVID). "
                "Run all methods in Compare mode — watch Gibbs stabilize while Rejection struggles."
            ),
        ),
        Scenario(
            name="② COVID-like with shortness of breath",
            evidence={"Fever": "Yes", "Cough": "Yes", "SOB": "Yes", "Fatigue": "Yes"},
            description=(
                "Respiratory symptoms (SOB is rare in prior). Rejection very low acceptance (~0.1%). "
                "Likelihood Weighting weights collapse due to multiple-parent collider. "
                "Gibbs excels: explores disease space efficiently. Compare mode shows the difference!"
            ),
        ),
        Scenario(
            name="③ Confirmed COVID + chest findings",
            evidence={"PCRTest": "Positive", "ChestXray": "Infiltrates", "Fever": "Yes", "SOB": "Yes"},
            description=(
                "Strong evidence: positive PCR (high specificity for COVID) + infiltrates. "
                "Rejection: ~0.01% acceptance. LW: OK but noisy. Gibbs: fastest convergence. "
                "Student should see P(COVID) → 0.95+, P(Severity=Severe) spike with age."
            ),
        ),
        Scenario(
            name="④ Ambiguous: fever + positive antigen (Flu or COVID?)",
            evidence={"AntigenTest": "Positive", "Fever": "Yes", "Cough": "Yes"},
            description=(
                "Ambiguous evidence: antigen positive but doesn't distinguish. "
                "Rejection OK acceptance (~0.5%). Prior → P(Flu), P(COVID) both moderate. "
                "Compare modes: see how methods differ when inference is genuinely uncertain."
            ),
        ),
    ],
)


ALL_MODELS = {
    "battery": BATTERY_MODEL,
    "fusion": SENSOR_FUSION_MODEL,
    "mission": MISSION_MODEL,
    "weather": WEATHER_MODEL,
    "robot_sampling": ROBOT_SAMPLING_MODEL,
    "student": STUDENT_MODEL,
    "ising": ISING_MODEL,
    "medical_diagnosis": MEDICAL_DIAGNOSIS_MODEL,
}
