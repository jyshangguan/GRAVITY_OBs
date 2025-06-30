# GRAVITY_OBs Documentation Website

This directory contains the documentation website for the GRAVITY_OBs package.

## Files

- `index.html` - Main documentation website
- `styles.css` - CSS styling for the website  
- `script.js` - JavaScript for interactive features

## Features

The documentation website includes:

- **Overview**: Package features and capabilities
- **Installation**: Step-by-step setup instructions
- **Module Documentation**: Comprehensive function listings for all modules
- **Examples**: Code examples for common use cases
- **Interactive Navigation**: Smooth scrolling and tabbed interface
- **Responsive Design**: Works on desktop and mobile devices
- **Copy Code**: Click to copy code examples
- **Dark Mode**: Toggle between light and dark themes

## Viewing the Documentation

### Option 1: Open Locally
Simply open `index.html` in your web browser.

### Option 2: Local Server
For better functionality (especially with external resources), serve with a local server:

```bash
# Using Python
python -m http.server 8000

# Using Node.js
npx http-server

# Using PHP
php -S localhost:8000
```

Then visit `http://localhost:8000` in your browser.

### Option 3: Deploy Online
Deploy to GitHub Pages, Netlify, or any static hosting service.

## Customization

- **Colors**: Modify the color scheme in `styles.css`
- **Content**: Update function descriptions in `index.html`
- **Examples**: Add more code examples in the examples section
- **Modules**: Add new modules by following the existing pattern

## Module Structure

The documentation covers these main modules:

1. **p2_tools**: ESO Phase 2 API integration and OB creation
2. **obs_utils**: Coordinate utilities and catalog access
3. **aspro_utils**: ASPRO compatibility tools
4. **colab_utils**: Google Colab integration
5. **gravidata**: GRAVITY data analysis tools

## Dependencies

The website uses these external resources:
- Font Awesome (icons)
- Prism.js (syntax highlighting)
- Modern CSS Grid and Flexbox (layout)

All dependencies are loaded from CDN, so an internet connection is required for full functionality.
