# AquiLLM Capstone Website

A simple main page for Django Capstone Website

## Structure

```
capstone_website/
├── main_app/              # Main Django app
│   ├── static/            # Static files (CSS, JS, images)
│   │   └── images/        # Logo and images
│   ├── templates/         # HTML templates
│   ├── views.py           # View functions
│   └── urls.py            # App-level URL routing
├── settings.py            # Django settings
├── urls.py                # Project-level URL routing
├── manage.py              # Django management script
└── README.md              # This file
```

## Setup (First Time Only)

1. Navigate to the capstone_website directory:
   ```bash
   cd capstone_website
   ```

2. Create a virtual environment (if not already created):
   ```bash
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

4. Install Django:
   ```bash
   pip install django
   ```

## How to Run

1. Make sure you're in the capstone_website directory and activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Run the Django development server:
   ```bash
   python3 manage.py runserver
   ```

3. Open your browser and go to:
   ```
   http://127.0.0.1:8000
   ```

4. When you're done, deactivate the virtual environment:
   ```bash
   deactivate
   ```

## What's Included

- Main landing page with AquiLLM logo
- Navigation bar with links to:
  - Team Members
  - Tools/Technologies
  - The Problem

## Notes

- The links in the nav bar are placeholders (#) and not functional yet
