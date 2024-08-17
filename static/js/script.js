document.addEventListener('DOMContentLoaded', () => {

    // Toggle icon navbar
    let menuIcon = document.querySelector('#menu-icon');
    let navbar = document.querySelector('.navbar');

    menuIcon.onclick = () => {
        menuIcon.classList.toggle('bx-x');
        navbar.classList.toggle('active');
    };

    // Scroll section active link and change header background color
    let sections = document.querySelectorAll('section');
    let navLinks = document.querySelectorAll('header nav a');
    let header = document.querySelector('.header');

    function updateHeaderColor() {
        sections.forEach(sec => {
            let top = window.scrollY;
            let offset = sec.offsetTop - 150;
            let height = sec.offsetHeight;
            let id = sec.getAttribute('id');

            if (top >= offset && top < offset + height) {
                switch (id) {
                    case 'home':
                        header.style.backgroundColor = 'var(--bg-color1)';
                        break;
                    case 'terms':
                        header.style.backgroundColor = 'var(--bg-color2)';
                        break;
                    case 'calculator':
                        header.style.backgroundColor = 'var(--bg-color3)';
                        break;
                    case 'about':
                        header.style.backgroundColor = 'var(--bg-color4)';
                        break;
                    case 'contact':
                        header.style.backgroundColor = 'var(--bg-color5)';
                        break;
                    default:
                        header.style.backgroundColor = 'var(--bg-color1)';
                        break;
                }

                // Update active class for navigation links
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    // Initial call to set header color based on current scroll position
    updateHeaderColor();

    // Event listener for scroll to update header color and active navigation link
    window.addEventListener('scroll', updateHeaderColor);

    // Sticky navbar
    header.classList.toggle('sticky', window.scrollY > 100);

    // Remove toggle icon and navbar active class when clicking navbar links
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            menuIcon.classList.remove('bx-x');
            navbar.classList.remove('active');
        });
    });

    // Scroll reveal initialization
    ScrollReveal({
        reset: true,
        distance: '80px',
        duration: 2000,
        delay: 200
    });

    ScrollReveal().reveal('.home-content, .heading', { origin: 'top' });
    ScrollReveal().reveal('.home-img, .contact-content, .contact-box, .calculator form', { origin: 'bottom' });
    ScrollReveal().reveal('.home-content h1, .about-img', { origin: 'left' });
    ScrollReveal().reveal('.home-content p, .about-content', { origin: 'right' });

});
function loadContent(url) {
    window.location.href = url;
}