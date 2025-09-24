const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

function toList(str, sep=",") {
  if (!str) return [];
  return str.split(sep).map(s => s.trim()).filter(Boolean);
}
function toLines(str) {
  if (!str) return [];
  return str.split(/\r?\n/).map(s => s.trim()).filter(Boolean);
}

function renderRecipeCard(r) {
  const scorePct = Math.max(0, Math.min(100, Math.round((r.score || 0)*100)));
  const badges = (r.badges || []).map(b => `<span class="badge">${b}</span>`).join(" ");
  const why = r.why || "";
  return `
    <div class="recipe">
      <h4>${r.title}</h4>
      <div class="progress" title="overall score"><i style="width:${scorePct}%"></i></div>
      <div class="badges">${badges}</div>
      <div class="meta">
        <div>Cal: ${r.calories ?? "—"}</div>
        <div>Carb: ${r.carbs_g ?? "—"} g</div>
        <div>Sugar: ${r.sugar_g ?? "—"} g</div>
        <div>Fiber: ${r.fiber_g ?? "—"} g</div>
        <div>Na: ${r.sodium_mg ?? "—"} mg</div>
        <div>Prep: ${r.prep_minutes ? r.prep_minutes+"m" : "—"}</div>
      </div>
      <p><b>Ingredients:</b> ${r.ingredients_text}</p>
      <p><b>Instructions:</b> ${r.instructions}</p>
      ${why ? `<p><b>Why it fits:</b> ${why}</p>` : ""}
    </div>
  `;
}

async function getRecommendations() {
  const conditions = $$("#resultsWrap input[name='cond']:checked").map(el => el.value);
  const allergies = toList($("#allergies").value);
  const dislikes = toList($("#dislikes").value);
  const pantry = toLines($("#pantry").value);
  const mustUsePantry = $("#mustUsePantry").checked;
  const minPantryPct = Math.max(0, Math.min(100, parseInt($("#minPantryPct").value || 0, 10)));
  const top_k = Math.max(1, Math.min(20, parseInt($("#topk").value || 10, 10)));

  const payload = {
    conditions, allergies, dislikes, pantry,
    top_k, must_use_pantry: mustUsePantry,
    min_pantry_coverage: minPantryPct / 100.0
  };

  $("#results").innerHTML = "Loading…";

  try {
    const res = await fetch("/api/recommend", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.error || "Unknown error");
    const cards = data.results.map(renderRecipeCard).join("");
    $("#results").innerHTML = cards || "No matches — try lowering filters or adding pantry items.";
  } catch (e) {
    $("#results").innerHTML = "Error: " + e.message;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  $("#goBtn").addEventListener("click", getRecommendations);
});
