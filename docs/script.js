// Mobile menu toggle
document.addEventListener('DOMContentLoaded', function() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            hamburger.classList.toggle('active');
        });
    }
    
    // Close menu when clicking on a link
    const navLinks = document.querySelectorAll('.nav-menu a');
    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            navMenu.classList.remove('active');
            hamburger.classList.remove('active');
        });
    });
});

// Module navigation
document.addEventListener('DOMContentLoaded', function() {
    const moduleButtons = document.querySelectorAll('.module-btn');
    const moduleContents = document.querySelectorAll('.module-content');
    
    moduleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetModule = this.getAttribute('data-module');
            
            // Remove active class from all buttons and contents
            moduleButtons.forEach(btn => btn.classList.remove('active'));
            moduleContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            const targetContent = document.getElementById(targetModule);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
});

// Example tabs navigation
document.addEventListener('DOMContentLoaded', function() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const exampleContents = document.querySelectorAll('.example-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetTab = this.getAttribute('data-tab');
            
            // Remove active class from all buttons and contents
            tabButtons.forEach(btn => btn.classList.remove('active'));
            exampleContents.forEach(content => content.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            this.classList.add('active');
            const targetContent = document.getElementById(targetTab);
            if (targetContent) {
                targetContent.classList.add('active');
            }
        });
    });
});

// Smooth scrolling for navigation links
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('a[href^="#"]');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href').substring(1);
            const targetElement = document.getElementById(targetId);
            
            if (targetElement) {
                const offsetTop = targetElement.offsetTop - 80; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
});

// Add active state to navigation based on scroll position
document.addEventListener('DOMContentLoaded', function() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('.nav-menu a[href^="#"]');
    
    function highlightNavigation() {
        let current = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop - 100;
            const sectionHeight = section.offsetHeight;
            
            if (window.scrollY >= sectionTop && window.scrollY < sectionTop + sectionHeight) {
                current = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${current}`) {
                link.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', highlightNavigation);
    highlightNavigation(); // Call once on load
});

// Add intersection observer for animations
document.addEventListener('DOMContentLoaded', function() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all feature cards and function cards
    const animatedElements = document.querySelectorAll('.feature-card, .function-card, .install-step, .requirements');
    animatedElements.forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Add copy functionality to code blocks
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('.code-block');
    
    codeBlocks.forEach(block => {
        // Create copy button
        const copyButton = document.createElement('button');
        copyButton.className = 'copy-button';
        copyButton.innerHTML = '<i class="fas fa-copy"></i>';
        copyButton.title = 'Copy code';
        
        // Position the button
        block.style.position = 'relative';
        copyButton.style.position = 'absolute';
        copyButton.style.top = '10px';
        copyButton.style.right = '10px';
        copyButton.style.background = '#00d4ff';
        copyButton.style.color = 'white';
        copyButton.style.border = 'none';
        copyButton.style.padding = '8px';
        copyButton.style.borderRadius = '4px';
        copyButton.style.cursor = 'pointer';
        copyButton.style.opacity = '0.7';
        copyButton.style.transition = 'opacity 0.3s ease';
        
        block.appendChild(copyButton);
        
        // Add hover effect
        copyButton.addEventListener('mouseenter', function() {
            this.style.opacity = '1';
        });
        
        copyButton.addEventListener('mouseleave', function() {
            this.style.opacity = '0.7';
        });
        
        // Add copy functionality
        copyButton.addEventListener('click', function() {
            const code = block.querySelector('code');
            const text = code.textContent;
            
            navigator.clipboard.writeText(text).then(function() {
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                copyButton.style.background = '#28a745';
                
                setTimeout(function() {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    copyButton.style.background = '#00d4ff';
                }, 2000);
            }).catch(function() {
                // Fallback for older browsers
                const textArea = document.createElement('textarea');
                textArea.value = text;
                document.body.appendChild(textArea);
                textArea.select();
                document.execCommand('copy');
                document.body.removeChild(textArea);
                
                copyButton.innerHTML = '<i class="fas fa-check"></i>';
                copyButton.style.background = '#28a745';
                
                setTimeout(function() {
                    copyButton.innerHTML = '<i class="fas fa-copy"></i>';
                    copyButton.style.background = '#00d4ff';
                }, 2000);
            });
        });
    });
});

// Add search functionality (basic implementation)
document.addEventListener('DOMContentLoaded', function() {
    // Add search input to navigation (optional)
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Search functions...';
    searchInput.className = 'search-input';
    searchInput.style.display = 'none'; // Hidden by default, can be shown if needed
    
    // Simple search functionality for function names
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const functionCards = document.querySelectorAll('.function-card');
        
        functionCards.forEach(card => {
            const functionName = card.querySelector('h4, h5');
            if (functionName) {
                const text = functionName.textContent.toLowerCase();
                if (text.includes(searchTerm) || searchTerm === '') {
                    card.style.display = 'block';
                } else {
                    card.style.display = 'none';
                }
            }
        });
    });
});

// Add dark mode toggle (optional feature)
document.addEventListener('DOMContentLoaded', function() {
    // Create dark mode toggle button
    const darkModeToggle = document.createElement('button');
    darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    darkModeToggle.className = 'dark-mode-toggle';
    darkModeToggle.title = 'Toggle dark mode';
    darkModeToggle.style.position = 'fixed';
    darkModeToggle.style.bottom = '20px';
    darkModeToggle.style.right = '20px';
    darkModeToggle.style.background = '#00d4ff';
    darkModeToggle.style.color = 'white';
    darkModeToggle.style.border = 'none';
    darkModeToggle.style.padding = '12px';
    darkModeToggle.style.borderRadius = '50%';
    darkModeToggle.style.cursor = 'pointer';
    darkModeToggle.style.boxShadow = '0 2px 10px rgba(0,0,0,0.2)';
    darkModeToggle.style.transition = 'all 0.3s ease';
    darkModeToggle.style.zIndex = '1000';
    
    document.body.appendChild(darkModeToggle);
    
    // Check for saved dark mode preference
    const isDarkMode = localStorage.getItem('darkMode') === 'true';
    if (isDarkMode) {
        document.body.classList.add('dark-mode');
        darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    darkModeToggle.addEventListener('click', function() {
        document.body.classList.toggle('dark-mode');
        const isDark = document.body.classList.contains('dark-mode');
        
        localStorage.setItem('darkMode', isDark);
        darkModeToggle.innerHTML = isDark ? '<i class="fas fa-sun"></i>' : '<i class="fas fa-moon"></i>';
    });
});
