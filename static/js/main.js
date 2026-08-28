document.addEventListener("DOMContentLoaded", () => {

    // ==============================
    // MOBILE NAVIGATION
    // ==============================

    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.querySelector(".nav-links");

    if (menuToggle && navLinks) {
        menuToggle.addEventListener("click", () => {
            navLinks.classList.toggle("active");
        });
    }


    // ==============================
    // FLASH MESSAGE CLOSE BUTTONS
    // ==============================

    const closeButtons = document.querySelectorAll(".flash-close");

    closeButtons.forEach((button) => {
        button.addEventListener("click", () => {
            const message = button.closest(".flash-message");

            if (message) {
                message.remove();
            }
        });
    });


    // ==============================
    // AUTO-HIDE FLASH MESSAGES
    // ==============================

    const flashMessages = document.querySelectorAll(".flash-message");

    flashMessages.forEach((message) => {
        setTimeout(() => {
            message.style.opacity = "0";
            message.style.transform = "translateX(20px)";

            setTimeout(() => {
                message.remove();
            }, 300);

        }, 4000);
    });


    // ==============================
    // FORM SUBMIT LOADING STATE
    // ==============================

    const predictionForm = document.querySelector(".prediction-form");

    if (predictionForm) {

        predictionForm.addEventListener("submit", () => {

            const button =
                predictionForm.querySelector(".predict-button");

            if (button) {

                button.innerHTML = `
                    <i class="bi bi-cpu"></i>
                    Analyzing...
                `;

                button.disabled = true;
            }

        });

    }

});