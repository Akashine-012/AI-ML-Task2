document.getElementById("prediction-form").addEventListener("submit", async function (e) {
    e.preventDefault();  // Prevent the form from submitting the default way

    // Collect input values
    const marketingSpend = document.getElementById("marketingSpend").value;
    const competitorPrice = document.getElementById("competitorPrice").value;
    const averagePrice = document.getElementById("averagePrice").value;
    const economicIndex = document.getElementById("economicIndex").value;
    const demographicFactor = document.getElementById("demographicFactor").value;
    const prescriptionRate = document.getElementById("prescriptionRate").value;
    const season = document.getElementById("season").value;

    // Prepare the season data
    let seasonSpring = 0, seasonSummer = 0, seasonWinter = 0;
    if (season === 'Spring') seasonSpring = 1;
    else if (season === 'Summer') seasonSummer = 1;
    else if (season === 'Winter') seasonWinter = 1;

    // Create a request body for the POST request
    const requestData = {
        MarketingSpend: parseFloat(marketingSpend),
        CompetitorPrice: parseFloat(competitorPrice),
        AveragePrice: parseFloat(averagePrice),
        EconomicIndex: parseFloat(economicIndex),
        DemographicFactor: parseFloat(demographicFactor),
        PrescriptionRate: parseFloat(prescriptionRate),
        Season_Spring: seasonSpring,
        Season_Summer: seasonSummer,
        Season_Winter: seasonWinter
    };

    try {
        // Make a POST request to the Flask backend
        const response = await fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestData)
        });

        // Get the response data
        const data = await response.json();
        
        // Display the prediction result
        const resultDiv = document.getElementById("result");
        if (data.prediction) {
            resultDiv.innerHTML = `<h2>Predicted Sales Volume: ${data.prediction.toFixed(2)}</h2>`;
        } else {
            resultDiv.innerHTML = `<h2>Error: ${data.error}</h2>`;
        }
    } catch (error) {
        console.error("Error occurred:", error);
        document.getElementById("result").innerHTML = `<h2>Failed to predict. Please check backend.</h2>`;
    }
});
