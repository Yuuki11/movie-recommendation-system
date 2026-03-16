const STATES = ["neutral", "like", "avoid"];
const STATE_LABELS = {
  neutral: "Neutral",
  like: "Like",
  avoid: "Avoid",
};

function renderChipState(chip, state) {
  const stateLabel = chip.querySelector(".chip-state");
  chip.dataset.state = state;
  if (stateLabel) {
    stateLabel.textContent = STATE_LABELS[state];
  }
}

function updateHiddenInputs() {
  const preferred = [];
  const avoided = [];

  document.querySelectorAll(".genre-chip").forEach((chip) => {
    const genre = chip.dataset.genre;
    const state = chip.dataset.state || "neutral";

    if (state === "like") {
      preferred.push(genre);
    }
    if (state === "avoid") {
      avoided.push(genre);
    }
  });

  const preferredInput = document.querySelector("#preferred_genres");
  const avoidedInput = document.querySelector("#avoided_genres");
  if (preferredInput) {
    preferredInput.value = preferred.join(",");
  }
  if (avoidedInput) {
    avoidedInput.value = avoided.join(",");
  }
}

function initializeGenreStates() {
  const grid = document.querySelector(".genre-grid");
  if (!grid) {
    return;
  }

  const preferred = new Set(
    (grid.dataset.preferred || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
  );
  const avoided = new Set(
    (grid.dataset.avoided || "")
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean),
  );

  document.querySelectorAll(".genre-chip").forEach((chip) => {
    const genre = chip.dataset.genre;
    let state = "neutral";
    if (preferred.has(genre)) {
      state = "like";
    }
    if (avoided.has(genre)) {
      state = "avoid";
    }

    chip.dataset.state = state;
    renderChipState(chip, state);
    chip.addEventListener("click", () => {
      const currentIndex = STATES.indexOf(chip.dataset.state || "neutral");
      renderChipState(chip, STATES[(currentIndex + 1) % STATES.length]);
      updateHiddenInputs();
    });
  });

  updateHiddenInputs();
}

function initializeRatingDisplay() {
  const slider = document.querySelector("#minimum_rating");
  const value = document.querySelector("#minimum_rating_value");
  if (!slider || !value) {
    return;
  }

  value.textContent = Number(slider.value).toFixed(1);
  slider.addEventListener("input", () => {
    value.textContent = Number(slider.value).toFixed(1);
  });
}

function clearSearchStateAfterRender() {
  const pageShell = document.querySelector(".page-shell");
  if (!pageShell || pageShell.dataset.hasResults !== "true") {
    return;
  }

  if (!window.location.search) {
    return;
  }

  window.history.replaceState({}, document.title, window.location.pathname);
}

document.addEventListener("DOMContentLoaded", () => {
  initializeGenreStates();
  initializeRatingDisplay();
  clearSearchStateAfterRender();
});
