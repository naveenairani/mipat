
//const pageOpenSound = new Audio('{{ url_for('static', filename='sounds/sound2.mp3') }}');

// Function to play hover sound
function playHoverSound() {
    hoverSound.currentTime = 0; // Reset sound to start
    hoverSound.play();
}

// Function to play button click sound
function playButtonClickSound() {
    buttonClickSound.currentTime = 0; // Reset sound to start
    buttonClickSound.play();
}

// Function to play page open sound and open a new page
function openNewPage(url) {
    pageOpenSound.currentTime = 0; // Reset sound to start
    pageOpenSound.play();
    setTimeout(() => {
        window.location.href = url; // Opens the new page in the same tab
    }, 500); // Delay to allow the sound to play before opening the new page
}
