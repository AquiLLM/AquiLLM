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

interface ProfileMemory {
  id: number;
  fact: string;
  revision: number;
  origin_label: string;
  source_label: string;
  reconciliation_status: string;
}

const UserSettings: React.FC = () => {
  const [themeSettings, setThemeSettings] = useState({
    color_scheme: 'aquillm_default_dark',
    font_family: 'verdana'
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [memories, setMemories] = useState<ProfileMemory[]>([]);
  const [memoryLoading, setMemoryLoading] = useState(true);
  const [memoryError, setMemoryError] = useState<string | null>(null);
  const [editingMemory, setEditingMemory] = useState<number | null>(null);
  const [replacement, setReplacement] = useState('');
  const [busyMemory, setBusyMemory] = useState<number | null>(null);
  const [reconciliation, setReconciliation] = useState<string[]>([]);

  const loadMemories = async () => {
    setMemoryError(null);
    try {
      const response = await fetch('/api/profile-memory/', { credentials: 'include', headers: { Accept: 'application/json' } });
      if (!response.ok) throw new Error('Unable to load profile memories.');
      const data = await response.json();
      setMemories(data.facts);
      setReconciliation(data.reconciliations.map((item: { reconciliation_status: string }) => item.reconciliation_status));
    } catch (err) {
      setMemoryError(err instanceof Error ? err.message : 'Unable to load profile memories.');
    } finally {
      setMemoryLoading(false);
    }
  };

  useEffect(() => { void loadMemories(); }, []);

  const changeMemory = async (memory: ProfileMemory, method: 'PATCH' | 'DELETE') => {
    setBusyMemory(memory.id);
    setMemoryError(null);
    try {
      const response = await fetch(`/api/profile-memory/${memory.id}/`, {
        method, credentials: 'include',
        headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCookie('csrftoken') },
        body: JSON.stringify(method === 'PATCH'
          ? { replacement, expected_revision: memory.revision }
          : { expected_revision: memory.revision }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Unable to update memory.');
      setMemories(current => method === 'DELETE'
        ? current.filter(item => item.id !== memory.id)
        : current.map(item => item.id === memory.id ? data.fact : item));
      setEditingMemory(null);
      setReplacement('');
      if (data.reconciliation_status === 'pending' || data.reconciliation_status === 'failed')
        setReconciliation(current => [...current, data.reconciliation_status]);
      // Reload also shows reconciliation for the erased, superseded version.
      await loadMemories();
    } catch (err) {
      setMemoryError(err instanceof Error ? err.message : 'Unable to update memory.');
    } finally {
      setBusyMemory(null);
    }
  };

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

      <section className="mb-8 p-6 bg-scheme-shade_3 border border-border-mid_contrast rounded-lg" aria-labelledby="profile-memory-heading">
        <h3 id="profile-memory-heading" className="text-lg font-semibold mb-2">Profile Memory</h3>
        <p className="mb-4">Manage remembered preferences and background.</p>
        <p className="mb-4">Deleting a memory may also hide older context whose origin cannot be verified.</p>
        {memoryError && <div role="alert">{memoryError}</div>}
        {memoryError && <button type="button" onClick={() => void loadMemories()}>Reload memories</button>}
        {memoryLoading ? <p>Loading memories...</p> : memories.length === 0 && <p>No active profile memories.</p>}
        {memories.map(memory => <article key={memory.id} className="mb-4 border rounded p-3">
          <p>{memory.fact}</p>
          <p className="text-text-dimmed">Origin: {memory.origin_label} · Source: {memory.source_label}</p>
          {editingMemory === memory.id ? <form onSubmit={event => { event.preventDefault(); void changeMemory(memory, 'PATCH'); }}>
            <label htmlFor={`memory-replacement-${memory.id}`}>Replacement for memory {memory.id}</label>
            <input id={`memory-replacement-${memory.id}`} value={replacement} onChange={event => setReplacement(event.target.value)} maxLength={2000} required className="border rounded p-2 w-full bg-scheme-shade_3" />
            <button type="submit" disabled={busyMemory !== null}>Save correction</button>
            <button type="button" onClick={() => setEditingMemory(null)} disabled={busyMemory !== null}>Cancel correction</button>
          </form> : <button type="button" aria-label={`Correct memory ${memory.id}`} disabled={busyMemory !== null} onClick={() => { setEditingMemory(memory.id); setReplacement(memory.fact); }}>Correct</button>}
          <button type="button" aria-label={`Delete memory ${memory.id}`} disabled={busyMemory !== null} onClick={() => void changeMemory(memory, 'DELETE')}>Delete</button>
        </article>)}
        {reconciliation.includes('pending') && <p role="status">External deletion pending.</p>}
        {reconciliation.includes('failed') && <p role="status">External deletion could not be confirmed.</p>}
      </section>

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
            className="bg-accent hover:bg-accent-dark text-text-normal py-2 px-4 rounded"
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
