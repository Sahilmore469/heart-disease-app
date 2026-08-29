const form = document.getElementById("predict-form");
const btn = document.getElementById("assess-btn");
const card = document.getElementById("readout-card");
const gaugeFill = document.getElementById("gauge-fill");
const gaugeNumber = document.getElementById("gauge-number");
const verdict = document.getElementById("readout-verdict");

const GAUGE_LENGTH = 251.3; // path length of the half-circle track

function verdictFor(pred, prob) {
  if (pred === 1) {
    if (prob >= 75) return "High risk — strongly suggests heart disease";
    return "Elevated risk — signs consistent with heart disease";
  }
  if (prob <= 25) return "Low risk — no strong indicators found";
  return "Low-moderate risk — mostly reassuring signs";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  btn.disabled = true;
  btn.textContent = "Analyzing…";
  card.classList.remove("risk-low", "risk-high");
  gaugeFill.style.strokeDashoffset = GAUGE_LENGTH;
  gaugeNumber.textContent = "--%";
  verdict.textContent = "Reading the chart…";

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!data.ok) throw new Error(data.error || "Prediction failed");

    const prob = data.probability;
    const pred = data.prediction;

    gaugeNumber.textContent = `${prob}%`;
    verdict.textContent = verdictFor(pred, prob);
    card.classList.add(pred === 1 ? "risk-high" : "risk-low");

    const offset = GAUGE_LENGTH * (1 - prob / 100);
    requestAnimationFrame(() => {
      gaugeFill.style.strokeDashoffset = offset;
    });
  } catch (err) {
    verdict.textContent = "Something went wrong analyzing that chart.";
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Assess risk";
  }
});
