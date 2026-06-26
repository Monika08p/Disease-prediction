// Theme toggle (dark/light), persisted in memory for the session
(function () {
    const root = document.documentElement;
    const toggleBtn = document.getElementById("theme-toggle");

    function applyTheme(theme) {
        root.setAttribute("data-theme", theme);
        if (toggleBtn) toggleBtn.textContent = theme === "dark" ? "🌙" : "☀️";
    }

    // Default to dark; respect OS preference if available
    const prefersLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
    applyTheme(prefersLight ? "light" : "dark");

    if (toggleBtn) {
        toggleBtn.addEventListener("click", function () {
            const current = root.getAttribute("data-theme");
            applyTheme(current === "dark" ? "light" : "dark");
        });
    }
})();

// Loading spinner on form submit
(function () {
    const form = document.getElementById("predict-form");
    if (!form) return;

    form.addEventListener("submit", function () {
        const btn = document.getElementById("predict-btn");
        const btnText = document.getElementById("btn-text");
        const spinner = document.getElementById("btn-spinner");
        if (btn) btn.disabled = true;
        if (btnText) btnText.textContent = "Predicting...";
        if (spinner) spinner.classList.remove("hidden");
    });
})();
