# Jekyll Site Setup

This directory now contains a Jekyll-powered site for displaying SAE feature analysis.

## Key Improvements

1. **Clean separation**: Content, layout, and data are separated
2. **Maintainable**: Edit about.md directly, no HTML needed  
3. **Data-driven**: Features are loaded from _data/features.yml
4. **Responsive**: Modern, mobile-friendly design
5. **GitHub Pages ready**: Works out of the box

## Local Development

```bash
jekyll serve
```

View at http://localhost:4000

## Updating Feature Data

```bash
python generate_features_data.py
```

## GitHub Pages Deployment

1. Push to GitHub repository
2. Settings > Pages > Deploy from branch > main > /docs
3. Site available at https://username.github.io/repository-name/ 