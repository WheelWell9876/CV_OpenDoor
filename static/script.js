// Start capturing for the door state (open/closed)
function startCapture(state) {
    fetch(`/start_capture/${state}`, {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.status);
        alert(data.status);
    })
    .catch(error => {
        console.error('Error starting capture:', error);
    });
}

// Stop capturing images
function stopCapture() {
    fetch('/stop_capture', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.status);
        alert(data.status);
    })
    .catch(error => {
        console.error('Error stopping capture:', error);
    });
}

function startTraining() {
    fetch('/start_training', {
        method: 'POST'
    })
    .then(response => response.json())
    .then(data => {
        alert(data.status);  // Training started message
    })
    .catch(error => {
        console.error('Error starting training:', error);
    });
}


