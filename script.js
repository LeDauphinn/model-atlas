const zoomData = [
  {
    type: "Core unit",
    title: "Weight",
    summary:
      "A weight is a learned number. It scales how strongly one input should matter.",
    what:
      "If an input is important, the model learns a larger positive or negative weight for it.",
    why:
      "Training mostly means adjusting lots of weights so the model becomes less wrong.",
    question: "How much should this signal influence the prediction?",
    bridge:
      "Combine many weighted inputs plus a bias, and you get a neuron.",
    formula: "prediction = w * x + b",
  },
  {
    type: "Small computation",
    title: "Neuron",
    summary:
      "A neuron mixes inputs with weights, adds a bias, then often applies an activation function.",
    what:
      "It is a tiny detector. Depending on its learned weights, it may react to a useful pattern.",
    why:
      "Neurons are the building blocks that let neural networks represent nonlinear relationships.",
    question:
      "Does this combination of signals look like the pattern I care about?",
    bridge:
      "Many neurons in parallel form a layer that extracts multiple patterns at once.",
    formula: "a = activation(w1x1 + w2x2 + ... + b)",
  },
  {
    type: "Feature extractor",
    title: "Layer",
    summary:
      "A layer is a collection of neurons that transform one representation into another.",
    what:
      "Early layers often capture simple patterns. Later layers combine them into more abstract ones.",
    why:
      "Layering lets models build complex ideas from simpler components.",
    question:
      "What new representation of the input would make prediction easier?",
    bridge:
      "Stack several layers and you get a network capable of deep feature learning.",
    formula: "h = activation(Wx + b)",
  },
  {
    type: "Learner",
    title: "Network",
    summary:
      "A network is a stack of layers whose parameters are jointly trained to minimize a loss.",
    what:
      "Given data, labels, a loss function, and an optimizer, the network gradually learns useful internal features.",
    why:
      "This is the core engine behind modern deep learning systems.",
    question:
      "How should all these weights move together so the whole model gets better?",
    bridge:
      "For language, one especially successful network design is the transformer.",
    formula: "weights <- weights - learning_rate * gradient",
  },
  {
    type: "Sequence engine",
    title: "Transformer",
    summary:
      "A transformer uses self-attention so each token can mix information from other tokens in context.",
    what:
      "Instead of processing tokens one by one, attention helps the model compare many relationships directly.",
    why:
      "It scales well, captures long-range dependencies, and became the dominant architecture for LLMs.",
    question:
      "Which parts of the context should this token pay attention to right now?",
    bridge:
      "Stack many transformer blocks, pretrain them on text, and you get a GPT-style model.",
    formula: "attention(Q, K, V) = softmax(QK^T / sqrt(d))V",
  },
  {
    type: "Full system",
    title: "GPT",
    summary:
      "A GPT is an autoregressive transformer trained to predict the next token from previous tokens.",
    what:
      "During generation, it repeatedly scores candidate next tokens, samples one, appends it, and continues.",
    why:
      "That simple training objective produces broad language abilities when scaled with enough data and compute.",
    question:
      "Given everything so far, what token is most plausible next?",
    bridge:
      "The same next-token loop powers chat, summarization, coding assistance, and many other language tasks.",
    formula: "P(text) = product of P(token_t | token_<t)",
  },
];

const codeExamples = {
  regression: {
    title: "Linear Regression with scikit-learn",
    description:
      "Use this when you want to predict a continuous number, such as house price, delivery time, or sales.",
    points: [
      "A train/test split checks whether the model generalizes.",
      "Mean squared error is a common regression metric.",
      "The coefficients show how the features influence the prediction.",
    ],
    code: `from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.DataFrame(
    {
        "hours_studied": [1, 2, 3, 4, 5, 6, 7, 8],
        "practice_tests": [0, 1, 1, 2, 2, 3, 3, 4],
        "score": [52, 55, 61, 67, 72, 76, 83, 88],
    }
)

X = data[["hours_studied", "practice_tests"]]
y = data["score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, predictions))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)`,
  },
  classification: {
    title: "Binary Classification with Logistic Regression",
    description:
      "Use this when the target is a category such as spam/not spam, churn/no churn, or fraud/not fraud.",
    points: [
      "The model outputs probabilities, not just hard labels.",
      "Scaling features often helps linear classifiers behave better.",
      "Accuracy alone is not enough when classes are imbalanced.",
    ],
    code: `from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pandas as pd

emails = pd.DataFrame(
    {
        "discount_words": [0, 1, 1, 0, 1, 0, 1, 0],
        "exclamation_marks": [0, 5, 3, 0, 6, 1, 4, 0],
        "is_spam": [0, 1, 1, 0, 1, 0, 1, 0],
    }
)

X = emails[["discount_words", "exclamation_marks"]]
y = emails["is_spam"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = Pipeline(
    [
        ("scale", StandardScaler()),
        ("clf", LogisticRegression()),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))`,
  },
  clustering: {
    title: "Customer Segmentation with K-Means",
    description:
      "Use clustering when you do not have labels and want to discover groups in the data.",
    points: [
      "K-means tries to place cluster centers so points are close to one center.",
      "You choose the number of clusters up front.",
      "Clusters can be useful for exploration, personalization, or anomaly triage.",
    ],
    code: `from sklearn.cluster import KMeans
import pandas as pd

customers = pd.DataFrame(
    {
        "monthly_spend": [25, 28, 31, 90, 95, 102, 180, 190, 210],
        "visits_per_month": [7, 6, 8, 4, 5, 4, 1, 2, 1],
    }
)

model = KMeans(n_clusters=3, random_state=42, n_init="auto")
customers["cluster"] = model.fit_predict(customers)

print(customers)
print("Cluster centers:")
print(model.cluster_centers_)`,
  },
  text: {
    title: "Text Classification with TF-IDF + Logistic Regression",
    description:
      "A lightweight NLP baseline often beats intuition. Start here before reaching for a large neural model.",
    points: [
      "TF-IDF turns text into numeric features based on word importance.",
      "A simple baseline gives you a performance floor to beat.",
      "Many real projects succeed with this setup when data is limited.",
    ],
    code: `from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

texts = [
    "I loved the course and learned a lot",
    "The explanations were confusing",
    "Fantastic project examples",
    "This lesson was difficult to follow",
    "Great pacing and practical demos",
    "I got lost halfway through",
]

labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.33, random_state=42
)

model = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))`,
  },
};

const attentionTokens = [
  "The",
  "model",
  "understood",
  "the",
  "question",
  "because",
  "attention",
  "connected",
  "the",
  "relevant",
  "words",
  ".",
];

const attentionWeights = [
  [0.28, 0.1, 0.04, 0.08, 0.1, 0.04, 0.08, 0.06, 0.07, 0.08, 0.05, 0.02],
  [0.08, 0.24, 0.11, 0.06, 0.13, 0.05, 0.12, 0.06, 0.05, 0.05, 0.03, 0.02],
  [0.03, 0.12, 0.22, 0.04, 0.14, 0.07, 0.13, 0.08, 0.04, 0.07, 0.04, 0.02],
  [0.08, 0.07, 0.04, 0.27, 0.1, 0.04, 0.07, 0.06, 0.08, 0.09, 0.06, 0.04],
  [0.04, 0.13, 0.12, 0.08, 0.19, 0.06, 0.1, 0.08, 0.05, 0.08, 0.05, 0.02],
  [0.04, 0.06, 0.06, 0.03, 0.08, 0.25, 0.13, 0.13, 0.03, 0.08, 0.07, 0.04],
  [0.02, 0.11, 0.13, 0.03, 0.09, 0.12, 0.22, 0.12, 0.03, 0.08, 0.04, 0.01],
  [0.03, 0.09, 0.1, 0.03, 0.08, 0.12, 0.2, 0.2, 0.03, 0.07, 0.04, 0.01],
  [0.08, 0.07, 0.04, 0.23, 0.09, 0.05, 0.08, 0.07, 0.11, 0.1, 0.06, 0.02],
  [0.03, 0.08, 0.08, 0.05, 0.11, 0.06, 0.11, 0.08, 0.05, 0.24, 0.08, 0.03],
  [0.03, 0.07, 0.07, 0.04, 0.1, 0.07, 0.11, 0.08, 0.04, 0.17, 0.19, 0.03],
  [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.45],
];

const toyTransitions = {
  start: [
    ["learns", 0.2],
    ["finds", 0.14],
    ["uses", 0.12],
    ["predicts", 0.18],
    ["patterns", 0.16],
    ["from", 0.2],
  ],
  machine: [["learning", 0.82], ["intelligence", 0.08], ["models", 0.1]],
  learning: [["models", 0.24], ["systems", 0.12], ["finds", 0.14], ["uses", 0.12], ["from", 0.18], ["patterns", 0.2]],
  models: [["learn", 0.28], ["predict", 0.18], ["compress", 0.1], ["use", 0.12], ["patterns", 0.18], ["well", 0.14]],
  model: [["uses", 0.28], ["predicts", 0.2], ["learns", 0.18], ["weights", 0.14], ["context", 0.2]],
  data: [["to", 0.16], ["patterns", 0.28], ["well", 0.08], ["into", 0.18], ["and", 0.15], ["from", 0.15]],
  transformers: [["use", 0.44], ["scale", 0.14], ["focus", 0.12], ["learn", 0.12], ["work", 0.18]],
  transformer: [["uses", 0.44], ["layers", 0.12], ["attention", 0.28], ["blocks", 0.16]],
  gpt: [["predicts", 0.28], ["continues", 0.16], ["samples", 0.18], ["tokens", 0.2], ["from", 0.18]],
  use: [["attention", 0.38], ["layers", 0.18], ["data", 0.14], ["context", 0.14], ["tokens", 0.16]],
  uses: [["attention", 0.34], ["weights", 0.12], ["layers", 0.16], ["context", 0.18], ["data", 0.2]],
  attention: [["to", 0.38], ["scores", 0.14], ["weights", 0.14], ["context", 0.2], ["patterns", 0.14]],
  tokens: [["to", 0.24], ["from", 0.16], ["with", 0.14], ["by", 0.08], ["efficiently", 0.1], ["well", 0.08], ["and", 0.2]],
  predicts: [["the", 0.24], ["next", 0.38], ["useful", 0.08], ["likely", 0.12], ["patterns", 0.18]],
  next: [["token", 0.72], ["step", 0.12], ["word", 0.08], ["move", 0.08]],
  token: [["from", 0.22], ["using", 0.18], ["with", 0.12], ["probabilities", 0.22], ["context", 0.26]],
  context: [["to", 0.18], ["well", 0.12], ["across", 0.18], ["into", 0.08], ["for", 0.12], ["tokens", 0.18], ["matters", 0.14]],
  patterns: [["in", 0.22], ["from", 0.18], ["across", 0.2], ["well", 0.1], ["and", 0.14], ["that", 0.16]],
  from: [["data", 0.32], ["context", 0.26], ["examples", 0.24], ["tokens", 0.18]],
  examples: [["and", 0.16], ["to", 0.18], ["well", 0.08], ["quickly", 0.12], ["that", 0.16], ["patterns", 0.3]],
  learn: [["patterns", 0.32], ["from", 0.26], ["features", 0.18], ["representations", 0.14], ["well", 0.1]],
  learns: [["patterns", 0.26], ["weights", 0.18], ["from", 0.24], ["useful", 0.12], ["representations", 0.2]],
  weights: [["that", 0.18], ["to", 0.12], ["from", 0.1], ["matter", 0.22], ["shape", 0.18], ["predictions", 0.2]],
  probabilities: [["and", 0.12], ["to", 0.16], ["before", 0.08], ["for", 0.08], ["each", 0.22], ["candidate", 0.34]],
  candidate: [["tokens", 0.48], ["choices", 0.22], ["outputs", 0.14], ["words", 0.16]],
  and: [["generalizes", 0.1], ["updates", 0.12], ["repeats", 0.16], ["then", 0.16], ["builds", 0.12], ["uses", 0.14], ["learns", 0.2]],
  the: [["next", 0.22], ["model", 0.12], ["loss", 0.08], ["context", 0.1], ["relevant", 0.14], ["data", 0.08], ["pattern", 0.12], ["weights", 0.14]],
};

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setupReveals() {
  if (prefersReducedMotion) {
    document.querySelectorAll(".reveal").forEach((el) => el.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
        }
      });
    },
    { threshold: 0.14 }
  );

  document.querySelectorAll(".reveal").forEach((el) => observer.observe(el));
}

function setupSpotlight() {
  const root = document.documentElement;
  window.addEventListener("pointermove", (event) => {
    const x = (event.clientX / window.innerWidth) * 100;
    const y = (event.clientY / window.innerHeight) * 100;
    root.style.setProperty("--spot-x", `${x}%`);
    root.style.setProperty("--spot-y", `${y}%`);
  });
}

function setupScrollProgress() {
  const progress = document.getElementById("scroll-progress");
  const update = () => {
    const scrollable = document.documentElement.scrollHeight - window.innerHeight;
    const ratio = scrollable > 0 ? window.scrollY / scrollable : 0;
    progress.style.width = `${Math.min(100, Math.max(0, ratio * 100))}%`;
  };
  update();
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
}

function setupZoomLadder() {
  const type = document.getElementById("zoom-type");
  const title = document.getElementById("zoom-title");
  const summary = document.getElementById("zoom-summary");
  const what = document.getElementById("zoom-what");
  const why = document.getElementById("zoom-why");
  const question = document.getElementById("zoom-question");
  const bridge = document.getElementById("zoom-bridge");
  const formula = document.getElementById("zoom-formula");
  const buttons = document.querySelectorAll(".zoom-step");

  const render = (index) => {
    const item = zoomData[index];
    type.textContent = item.type;
    title.textContent = item.title;
    summary.textContent = item.summary;
    what.textContent = item.what;
    why.textContent = item.why;
    question.textContent = item.question;
    bridge.textContent = item.bridge;
    formula.textContent = item.formula;

    buttons.forEach((button) => {
      button.classList.toggle("is-active", Number(button.dataset.step) === index);
    });
  };

  buttons.forEach((button) => {
    button.addEventListener("click", () => render(Number(button.dataset.step)));
  });

  render(0);
}

function setupConstellation() {
  const canvas = document.getElementById("constellation");
  const ctx = canvas.getContext("2d");
  let particles = [];

  const resize = () => {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    const count = Math.min(78, Math.floor(window.innerWidth / 18));
    particles = Array.from({ length: count }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 1.8 + 1,
    }));
  };

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < particles.length; i += 1) {
      const particle = particles[i];
      particle.x += particle.vx;
      particle.y += particle.vy;

      if (particle.x < 0 || particle.x > canvas.width) {
        particle.vx *= -1;
      }
      if (particle.y < 0 || particle.y > canvas.height) {
        particle.vy *= -1;
      }

      ctx.beginPath();
      ctx.arc(particle.x, particle.y, particle.r, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(23, 139, 142, 0.28)";
      ctx.fill();

      for (let j = i + 1; j < particles.length; j += 1) {
        const other = particles[j];
        const dx = particle.x - other.x;
        const dy = particle.y - other.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        if (distance < 110) {
          ctx.beginPath();
          ctx.moveTo(particle.x, particle.y);
          ctx.lineTo(other.x, other.y);
          ctx.strokeStyle = `rgba(19, 42, 56, ${(1 - distance / 110) * 0.16})`;
          ctx.stroke();
        }
      }
    }

    if (!prefersReducedMotion) {
      window.requestAnimationFrame(draw);
    }
  };

  resize();
  if (prefersReducedMotion) {
    draw();
  } else {
    window.requestAnimationFrame(draw);
  }
  window.addEventListener("resize", resize);
}

function setupRegressionLab() {
  const canvas = document.getElementById("regression-canvas");
  const ctx = canvas.getContext("2d");

  const slopeSlider = document.getElementById("slope-slider");
  const interceptSlider = document.getElementById("intercept-slider");
  const learningRateSlider = document.getElementById("learning-rate-slider");
  const noiseSlider = document.getElementById("noise-slider");
  const mseValue = document.getElementById("mse-value");
  const slopeValue = document.getElementById("slope-value");
  const interceptValue = document.getElementById("intercept-value");
  const targetLine = document.getElementById("target-line");

  let model = { m: Number(slopeSlider.value), b: Number(interceptSlider.value) };
  let target = { m: 1.8, b: 0.3 };
  let noise = Number(noiseSlider.value);
  let dataset = [];
  let trainingTimer = null;

  const createDataset = () => {
    target = {
      m: (Math.random() * 3.4 - 1.7).toFixed(2),
      b: (Math.random() * 1.8 - 0.9).toFixed(2),
    };
    dataset = Array.from({ length: 28 }, (_, index) => {
      const x = -3 + (index / 27) * 6;
      const y =
        Number(target.m) * x +
        Number(target.b) +
        (Math.random() - 0.5) * 2 * noise;
      return { x, y };
    });
    targetLine.textContent = `m=${Number(target.m).toFixed(2)}, b=${Number(target.b).toFixed(2)}`;
  };

  const worldToCanvas = (x, y) => {
    const padding = 42;
    const plotWidth = canvas.width - padding * 2;
    const plotHeight = canvas.height - padding * 2;
    const cx = padding + ((x + 3.5) / 7) * plotWidth;
    const cy = canvas.height - padding - ((y + 4) / 8) * plotHeight;
    return { x: cx, y: cy };
  };

  const predict = (x) => model.m * x + model.b;

  const mse = () =>
    dataset.reduce((sum, point) => {
      const error = predict(point.x) - point.y;
      return sum + error * error;
    }, 0) / dataset.length;

  const gradientStep = () => {
    const lr = Number(learningRateSlider.value);
    const n = dataset.length;
    let dm = 0;
    let db = 0;

    dataset.forEach((point) => {
      const error = predict(point.x) - point.y;
      dm += (2 / n) * error * point.x;
      db += (2 / n) * error;
    });

    model.m -= lr * dm;
    model.b -= lr * db;
    slopeSlider.value = model.m.toFixed(2);
    interceptSlider.value = model.b.toFixed(2);
    draw();
  };

  const drawAxes = () => {
    ctx.strokeStyle = "rgba(19, 42, 56, 0.18)";
    ctx.lineWidth = 1;
    const xAxisStart = worldToCanvas(-3.5, 0);
    const xAxisEnd = worldToCanvas(3.5, 0);
    const yAxisStart = worldToCanvas(0, -4);
    const yAxisEnd = worldToCanvas(0, 4);

    ctx.beginPath();
    ctx.moveTo(xAxisStart.x, xAxisStart.y);
    ctx.lineTo(xAxisEnd.x, xAxisEnd.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(yAxisStart.x, yAxisStart.y);
    ctx.lineTo(yAxisEnd.x, yAxisEnd.y);
    ctx.stroke();
  };

  const drawLine = (m, b, strokeStyle, width, dash = []) => {
    const p1 = worldToCanvas(-3.5, m * -3.5 + b);
    const p2 = worldToCanvas(3.5, m * 3.5 + b);
    ctx.save();
    ctx.strokeStyle = strokeStyle;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    ctx.moveTo(p1.x, p1.y);
    ctx.lineTo(p2.x, p2.y);
    ctx.stroke();
    ctx.restore();
  };

  const draw = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawAxes();

    dataset.forEach((point) => {
      const p = worldToCanvas(point.x, point.y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(23, 139, 142, 0.82)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255, 255, 255, 0.88)";
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    drawLine(Number(target.m), Number(target.b), "rgba(217, 168, 60, 0.92)", 3, [8, 8]);
    drawLine(model.m, model.b, "rgba(230, 95, 61, 0.96)", 4);

    mseValue.textContent = mse().toFixed(3);
    slopeValue.textContent = model.m.toFixed(2);
    interceptValue.textContent = model.b.toFixed(2);
  };

  const stopTraining = () => {
    if (trainingTimer) {
      window.clearInterval(trainingTimer);
      trainingTimer = null;
    }
  };

  document.getElementById("step-train").addEventListener("click", () => {
    stopTraining();
    gradientStep();
  });

  document.getElementById("auto-train").addEventListener("click", () => {
    stopTraining();
    let steps = 0;
    trainingTimer = window.setInterval(() => {
      gradientStep();
      steps += 1;
      if (steps >= 60) {
        stopTraining();
      }
    }, prefersReducedMotion ? 0 : 50);
  });

  document.getElementById("reset-model").addEventListener("click", () => {
    stopTraining();
    model = { m: -0.8, b: 1.2 };
    slopeSlider.value = String(model.m);
    interceptSlider.value = String(model.b);
    draw();
  });

  document.getElementById("new-dataset").addEventListener("click", () => {
    stopTraining();
    noise = Number(noiseSlider.value);
    createDataset();
    draw();
  });

  slopeSlider.addEventListener("input", (event) => {
    stopTraining();
    model.m = Number(event.target.value);
    draw();
  });

  interceptSlider.addEventListener("input", (event) => {
    stopTraining();
    model.b = Number(event.target.value);
    draw();
  });

  noiseSlider.addEventListener("input", (event) => {
    noise = Number(event.target.value);
  });

  createDataset();
  draw();
}

function setupNetworkLab() {
  const freeInput = document.getElementById("feature-free");
  const urgentInput = document.getElementById("feature-urgent");
  const freeValue = document.getElementById("free-value");
  const urgentValue = document.getElementById("urgent-value");
  const outputValue = document.getElementById("network-output");
  const verdict = document.getElementById("network-verdict");

  const hiddenNodes = [
    {
      weights: [1.5, 0.35],
      bias: -0.4,
      valueEl: document.getElementById("hidden-0-value"),
      barEl: document.getElementById("hidden-0-bar"),
    },
    {
      weights: [0.35, 1.6],
      bias: -0.5,
      valueEl: document.getElementById("hidden-1-value"),
      barEl: document.getElementById("hidden-1-bar"),
    },
    {
      weights: [-1.1, -0.8],
      bias: 1.0,
      valueEl: document.getElementById("hidden-2-value"),
      barEl: document.getElementById("hidden-2-bar"),
    },
  ];
  const outputWeights = [1.6, 1.2, -1.1];
  const outputBias = -0.25;

  const relu = (value) => Math.max(0, value);
  const sigmoid = (value) => 1 / (1 + Math.exp(-value));

  const update = () => {
    const inputs = [Number(freeInput.value), Number(urgentInput.value)];
    freeValue.textContent = inputs[0].toFixed(2);
    urgentValue.textContent = inputs[1].toFixed(2);

    const hiddenActivations = hiddenNodes.map((node) => {
      const weightedSum =
        node.weights[0] * inputs[0] + node.weights[1] * inputs[1] + node.bias;
      const activation = relu(weightedSum);
      node.valueEl.textContent = activation.toFixed(2);
      node.barEl.style.width = `${Math.min(100, activation * 65)}%`;
      return activation;
    });

    const logit =
      hiddenActivations.reduce(
        (sum, activation, index) => sum + activation * outputWeights[index],
        outputBias
      );
    const probability = sigmoid(logit);
    outputValue.textContent = probability.toFixed(2);
    verdict.textContent =
      probability >= 0.5 ? "Likely spam message" : "Likely safe email";
  };

  freeInput.addEventListener("input", update);
  urgentInput.addEventListener("input", update);
  update();
}

function approximateTokenize(text) {
  const pieces = text
    .trim()
    .match(/[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]/g);

  if (!pieces) {
    return [];
  }

  const tokens = [];

  pieces.forEach((piece) => {
    if (!/[A-Za-z]/.test(piece) || piece.length <= 4) {
      tokens.push({ value: piece, subword: false });
      return;
    }

    const lower = piece.toLowerCase();
    if (lower.endsWith("ing") && piece.length > 6) {
      tokens.push({ value: piece.slice(0, -3), subword: false });
      tokens.push({ value: "##ing", subword: true });
      return;
    }

    if (lower.endsWith("tion") && piece.length > 7) {
      tokens.push({ value: piece.slice(0, -4), subword: false });
      tokens.push({ value: "##tion", subword: true });
      return;
    }

    if (piece.length > 9) {
      tokens.push({ value: piece.slice(0, 4), subword: false });
      let rest = piece.slice(4);
      while (rest.length > 0) {
        tokens.push({ value: `##${rest.slice(0, 3)}`, subword: true });
        rest = rest.slice(3);
      }
      return;
    }

    if (piece.length > 6) {
      tokens.push({ value: piece.slice(0, 3), subword: false });
      tokens.push({ value: `##${piece.slice(3)}`, subword: true });
      return;
    }

    tokens.push({ value: piece, subword: false });
  });

  return tokens;
}

function setupTokenLab() {
  const input = document.getElementById("token-input");
  const output = document.getElementById("token-output");
  const tokenCount = document.getElementById("token-count");
  const characterCount = document.getElementById("character-count");

  const render = () => {
    const text = input.value;
    const tokens = approximateTokenize(text);
    output.innerHTML = "";

    tokens.forEach((token) => {
      const chip = document.createElement("span");
      chip.className = `token-chip${token.subword ? " subword" : ""}`;
      chip.textContent = token.value;
      output.appendChild(chip);
    });

    tokenCount.textContent = String(tokens.length);
    characterCount.textContent = String(text.length);
  };

  input.addEventListener("input", render);
  render();
}

function setupAttentionLab() {
  const sentence = document.getElementById("attention-sentence");
  const bars = document.getElementById("attention-bars");
  const caption = document.getElementById("attention-caption");
  let focusIndex = 6;

  const render = () => {
    sentence.innerHTML = "";
    bars.innerHTML = "";

    attentionTokens.forEach((token, index) => {
      const button = document.createElement("button");
      button.className = `attention-token${index === focusIndex ? " is-focus" : ""}`;
      button.textContent = token;
      button.style.opacity = String(0.45 + attentionWeights[focusIndex][index] * 1.4);
      button.addEventListener("click", () => {
        focusIndex = index;
        render();
      });
      sentence.appendChild(button);
    });

    caption.textContent = `Focus token "${attentionTokens[focusIndex]}" pays the most attention to the tokens with the tallest bars.`;

    attentionWeights[focusIndex].forEach((weight, index) => {
      const row = document.createElement("div");
      row.className = "attention-bar-row";
      row.innerHTML = `
        <strong>${attentionTokens[index]}</strong>
        <div class="attention-bar-track"><span style="width:${(weight * 100).toFixed(1)}%"></span></div>
        <span>${weight.toFixed(2)}</span>
      `;
      bars.appendChild(row);
    });
  };

  render();
}

function normalizeDistribution(entries, temperature) {
  const adjusted = entries.map(([token, weight]) => [
    token,
    Math.exp(Math.log(weight) / temperature),
  ]);
  const total = adjusted.reduce((sum, [, weight]) => sum + weight, 0);
  return adjusted.map(([token, weight]) => [token, weight / total]);
}

function weightedSample(entries) {
  let threshold = Math.random();
  for (const [token, probability] of entries) {
    threshold -= probability;
    if (threshold <= 0) {
      return token;
    }
  }
  return entries[entries.length - 1][0];
}

function chooseState(text) {
  const tokens = text
    .trim()
    .toLowerCase()
    .match(/[a-z]+/g);

  if (!tokens || tokens.length === 0) {
    return "start";
  }

  const last = tokens[tokens.length - 1];
  if (toyTransitions[last]) {
    return last;
  }

  if (tokens.includes("transformer") || tokens.includes("transformers")) {
    return tokens.includes("transformers") ? "transformers" : "transformer";
  }
  if (tokens.includes("gpt")) {
    return "gpt";
  }
  if (tokens.includes("machine")) {
    return "machine";
  }
  if (tokens.includes("data")) {
    return "data";
  }
  return "start";
}

function setupGptLab() {
  const promptInput = document.getElementById("gpt-prompt");
  const temperatureSlider = document.getElementById("temperature-slider");
  const generatedText = document.getElementById("generated-text");
  const probabilityBars = document.getElementById("probability-bars");
  const generateOne = document.getElementById("generate-token");
  const generateSequence = document.getElementById("generate-sequence");
  const resetButton = document.getElementById("reset-generation");
  let generated = promptInput.value.trim();

  const renderDistribution = () => {
    const state = chooseState(generated);
    const base = toyTransitions[state] || toyTransitions.start;
    const distribution = normalizeDistribution(base, Number(temperatureSlider.value));
    probabilityBars.innerHTML = "";

    distribution.forEach(([token, probability]) => {
      const row = document.createElement("div");
      row.className = "probability-bar";
      row.innerHTML = `
        <strong>${token}</strong>
        <div class="probability-track"><span style="width:${(probability * 100).toFixed(1)}%"></span></div>
        <span>${(probability * 100).toFixed(0)}%</span>
      `;
      probabilityBars.appendChild(row);
    });
  };

  const renderText = () => {
    generatedText.textContent = generated || "(empty prompt)";
    renderDistribution();
  };

  const addToken = () => {
    const state = chooseState(generated);
    const distribution = normalizeDistribution(
      toyTransitions[state] || toyTransitions.start,
      Number(temperatureSlider.value)
    );
    const token = weightedSample(distribution);
    generated = generated ? `${generated} ${token}` : token;
    renderText();
  };

  promptInput.addEventListener("input", () => {
    generated = promptInput.value.trim();
    renderText();
  });

  temperatureSlider.addEventListener("input", renderDistribution);
  generateOne.addEventListener("click", addToken);
  generateSequence.addEventListener("click", () => {
    for (let step = 0; step < 6; step += 1) {
      addToken();
    }
  });
  resetButton.addEventListener("click", () => {
    generated = promptInput.value.trim();
    renderText();
  });

  renderText();
}

function setupCodeStudio() {
  const tabs = document.querySelectorAll(".code-tab");
  const title = document.getElementById("code-title");
  const description = document.getElementById("code-description");
  const points = document.getElementById("code-points");
  const block = document.getElementById("code-block");

  const render = (key) => {
    const example = codeExamples[key];
    title.textContent = example.title;
    description.textContent = example.description;
    block.textContent = example.code;
    points.innerHTML = "";
    example.points.forEach((point) => {
      const item = document.createElement("li");
      item.textContent = point;
      points.appendChild(item);
    });

    tabs.forEach((tab) => {
      tab.classList.toggle("is-active", tab.dataset.code === key);
    });
  };

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => render(tab.dataset.code));
  });

  render("regression");
}

function setupQuiz() {
  document.querySelectorAll(".quiz-card").forEach((card) => {
    const answer = card.dataset.answer;
    const feedback = card.querySelector(".quiz-feedback");

    card.querySelectorAll("button").forEach((button) => {
      button.addEventListener("click", () => {
        const correct = button.dataset.choice === answer;
        card.classList.toggle("is-correct", correct);
        card.classList.toggle("is-wrong", !correct);
        feedback.textContent = correct
          ? "Correct. That is the core idea."
          : `Not quite. The right answer is "${answer}".`;
      });
    });
  });
}

setupReveals();
setupSpotlight();
setupScrollProgress();
setupZoomLadder();
setupConstellation();
setupRegressionLab();
setupNetworkLab();
setupTokenLab();
setupAttentionLab();
setupGptLab();
setupCodeStudio();
setupQuiz();
