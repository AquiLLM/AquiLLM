// src/components/ThemeSelector.js
import React, { useState, useEffect } from 'react';
import { getCookie } from '../utils/csrf';

const themeDescriptions: Record<string, string> = {
  aquillm_default_dark:
    'A sleek, dark‑mode palette with deep charcoal backgrounds and subtle accent colors for low‑light environments. The blue you see comes from the color the star Altair primarily emits according to its spectrogram, which is the brightest star in the constellation Aquila, depicted in our logo. The orange, in contrast yet complement, is the color Altair emits the least of.',
  aquillm_default_light:
    'A clean, light color scheme with white backgrounds and gentle grays—perfect for well‑lit desks. The blue you see comes from the color the star Altair primarily emits according to its spectrogram, which is the brightest star in the constellation Aquila, depicted in our logo. The orange, in contrast yet complement, is the color Altair emits the least of.',
  aquillm_default_dark_accessible_chat:
    'Dark mode optimized for the chat page that omits the full blue and orange background of chat bubbles in favor of better readability.',
  aquillm_default_light_accessible_chat:
    'Light mode optimized for the chat page that omits the full blue and orange background of chat bubbles in favor of better readability.',
  high_contrast:
    'Maximum contrast everywhere: bold borders, stark black‑on‑white text, and clear focus indicators.',
};

const UserSettings: React.FC = () => {
  const [themeSettings, setThemeSettings] = useState({
    color_scheme: 'aquillm_default_dark',
    font_family: 'verdana'
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch current settings on mount.
  useEffect(() => {
    fetch('/api/user-settings/', {
      credentials: 'include', // ensure cookies are sent
      headers: { 'Accept': 'application/json' } // force a JSON response
    })
      .then(res => res.json())
      .then(data => {
        console.log('Fetched settings:', data); // Check the structure!
        setThemeSettings(data);
        setLoading(false);
      })
      .catch(err => {
        console.error('Error fetching user settings:', err);
        setError(err);
        setLoading(false);
      });
  }, []);

// Update state when user selects a new value
const handleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const { name, value } = e.target;
    setThemeSettings((prev) => ({ ...prev, [name]: value }));
};

  // Submit the changes to the server.
interface SubmitResponse {
    color_scheme: string;
    font_family: string;
}

const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    fetch('/api/user-settings/', {
      method: 'POST', 
      headers: { 
        'Content-Type': 'application/json',
        'X-CSRFToken': getCookie('csrftoken'),
      },
      credentials: 'include',
      body: JSON.stringify(themeSettings)
    })
      .then(res => res.json())
      .then((data: SubmitResponse) => {
        applyTheme(data);
      })
      .catch((err: Error) => console.error('Error updating settings:', err));
};

  // This helper function applies the chosen theme to the page.
  const applyTheme = ({ color_scheme, font_family }: { color_scheme: string; font_family: string }) => {
    // Remove any previously set theme classes (if using utility classes to indicate theme).
    document.body.classList.remove('theme-aquillm_default_dark', 'theme-aquillm_default_light', 'theme-aquillm_default_light_accessible_chat', 'theme-aquillm_default_dark_accessible_chat', 'theme-high_contrast');
    document.body.classList.add(`theme-${color_scheme}`);

    // If you want to change the font globally, you could update a CSS variable or
    // attach a class to <html> or <body> that Tailwind uses to control typography.
    document.body.classList.remove('font-latin_modern_roman', 'font-sans_serif', 'font-lexend', 'font-comicsans', 'font-opendyslexic', 'font-timesnewroman', 'font-verdana');
    document.body.classList.add(`font-${font_family}`);
  };

  if (loading) return <div>Loading settings...</div>;
  if (error) return <div>Error loading settings.</div>;

  return (
    <div className="p-[30px]">
      <h2 className="mb-6 text-2xl font-bold">User Settings</h2>

      {/* Zotero Integration Section */}
      <div className="mb-8 p-6 bg-scheme-shade_3 border border-border-mid_contrast rounded-lg">
        <h3 className="text-lg font-semibold mb-2">Zotero Integration</h3>
        <p className="text-text-dimmed mb-4">Connect your Zotero account to sync your research library with AquiLLM</p>
        <a
          href="/zotero/settings/"
          className="inline-block bg-red-600 hover:bg-red-700 text-white font-semibold py-2 px-6 rounded transition-colors"
        >
          Manage Zotero Connection
        </a>
      </div>

      <h2 className="mb-4 text-xl font-bold">Select Your Theme</h2>
      {/* Flex container: form on left, description on right */}
      <form onSubmit={handleSubmit} className="flex">
        {/* Left column: controls */}
        <div className="w-1/2 pr-4">
          <div className="mb-4">
            <label htmlFor="color_scheme" className="block mb-1">
              Color Scheme:
            </label>
            <select
              id="color_scheme"
              name="color_scheme"
              value={themeSettings.color_scheme}
              onChange={handleChange}
              className="border p-2 rounded w-full bg-scheme-shade_3 text-text-normal"
            >
              <option value="aquillm_default_dark">
                Aquillm Default Dark
              </option>
              <option value="aquillm_default_light">
                Aquillm Default Light
              </option>
              <option value="aquillm_default_dark_accessible_chat">
                Aquillm Default Dark – Accessible Chat
              </option>
              <option value="aquillm_default_light_accessible_chat">
                Aquillm Default Light – Accessible Chat
              </option>
              <option value="high_contrast">High Contrast</option>
            </select>
          </div>
          <div className="mb-4">
            <label htmlFor="font_family" className="block mb-1">
              Font Family:
            </label>
            <select
              id="font_family"
              name="font_family"
              value={themeSettings.font_family}
              onChange={handleChange}
              className="border p-2 rounded w-full bg-scheme-shade_3 text-text-normal"
            >
              <option value="sans_serif">Sans‑serif (Default)</option>
              <option value="verdana">Verdana</option>
              <option value="timesnewroman">Times New Roman</option>
              <option value="opendyslexic">Open Dyslexic</option>
              <option value="lexend">Lexend</option>
              <option value="comicsans">Comic Sans (Helps with dyslexia!)</option>
            </select>
          </div>
          <button
            type="submit"
            data-testid="save-theme-settings"
            className="bg-blue-600 text-white py-2 px-4 rounded"
          >
            Save Settings
          </button>
        </div>

        {/* Right column: description */}
        <div className="w-1/2 pl-4 border-l">
          <h3 className="font-semibold mb-2">Theme Description</h3>
          <p>{themeDescriptions[themeSettings.color_scheme]}</p>
        </div>
      </form>
    </div>
  );
};

export default UserSettings;