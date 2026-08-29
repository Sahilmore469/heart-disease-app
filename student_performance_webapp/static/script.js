const form = document.getElementById("predict-form");
const btn = document.getElementById("grade-btn");
const scoreWrap = document.getElementById("score-wrap");
const scoreNumber = document.getElementById("score-number");
const remark = document.getElementById("report-remark");

function remarkFor(score) {
  if (score >= 90) return "Outstanding work! 🌟";
  if (score >= 80) return "Great job — strong grasp.";
  if (score >= 70) return "Solid, steady progress.";
  if (score >= 60) return "Passing — some room to grow.";
  return "Needs support — let's dig in.";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  btn.disabled = true;
  btn.textContent = "Grading…";
  scoreWrap.classList.remove("graded");
  scoreNumber.textContent = "--";
  remark.textContent = "Reading the sheet…";

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

    scoreNumber.textContent = data.prediction;
    remark.textContent = remarkFor(data.prediction);

    // restart the draw-circle animation
    void scoreWrap.offsetWidth;
    scoreWrap.classList.add("graded");
  } catch (err) {
    remark.textContent = "Something went wrong grading that sheet.";
    console.error(err);
  } finally {
    btn.disabled = false;
    btn.textContent = "Grade this student";
  }
});
