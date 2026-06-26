// Renders the risk gauge (doughnut chart styled as a gauge) and the
// top contributing factors bar chart on the prediction result page.
// Expects global vars riskProba, riskLabel, factorLabels, factorScores
// to be set inline by predict.html before this script loads.

document.addEventListener("DOMContentLoaded", function () {
    if (typeof riskProba === "undefined") return;

    const riskColor = riskLabel === "High Risk" ? "#ff5f6d"
                     : riskLabel === "Medium Risk" ? "#ffc44d"
                     : "#3ddc84";

    // --- Risk gauge ---
    const gaugeCtx = document.getElementById("riskGauge");
    if (gaugeCtx) {
        new Chart(gaugeCtx, {
            type: "doughnut",
            data: {
                datasets: [{
                    data: [riskProba, 100 - riskProba],
                    backgroundColor: [riskColor, "rgba(150,150,150,0.15)"],
                    borderWidth: 0,
                }],
            },
            options: {
                cutout: "75%",
                rotation: -90,
                circumference: 180,
                plugins: { legend: { display: false }, tooltip: { enabled: false } },
            },
        });
    }

    // --- Top contributing factors bar chart ---
    const factorsCtx = document.getElementById("factorsChart");
    if (factorsCtx && typeof factorLabels !== "undefined") {
        new Chart(factorsCtx, {
            type: "bar",
            data: {
                labels: factorLabels,
                datasets: [{
                    label: "Contribution Score",
                    data: factorScores,
                    backgroundColor: riskColor,
                    borderRadius: 6,
                }],
            },
            options: {
                indexAxis: "y",
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { color: "rgba(150,150,150,0.1)" } },
                    y: { grid: { display: false } },
                },
            },
        });
    }
});
