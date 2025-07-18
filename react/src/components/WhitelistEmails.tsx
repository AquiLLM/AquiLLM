import React, { useState, useEffect } from 'react';
import { Trash2 } from 'lucide-react';
import formatUrl from '../utils/formatUrl';
import { getCsrfCookie } from '../main';

const WhitelistEmails: React.FC = () => {
  const [emails, setEmails] = useState<string[]>([]);
  const [newEmail, setNewEmail] = useState<string>('');
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [inputError, setInputError] = useState<string | null>(null);

  // Fetch emails on component mount
  useEffect(() => {
    const fetchEmails = async () => {
      try {
        setLoading(true);
        const response = await fetch(window.apiUrls.api_whitelist_emails);
        if (!response.ok) {
          throw new Error('Failed to fetch emails');
        }
        const data: string[] = (await response.json()).whitelisted;
        setEmails(data);
      } catch (err) {
        console.error(err);
        setError('Error fetching emails');
      } finally {
        setLoading(false);
      }
    };

    fetchEmails();
  }, []);

  // Reset input error when newEmail changes
  useEffect(() => {
    setInputError(null);
  }, [newEmail]);

  // Validate email format
  const isValidEmail = (email: string) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // Add a new email
  const handleAddEmail = async () => {
    const trimmedEmail = newEmail.trim();
    if (!isValidEmail(trimmedEmail)) {
      setInputError('Invalid email address');
      return;
    }
    if (emails.includes(trimmedEmail)) {
      setInputError('Email already whitelisted');
      return;
    }
    setInputError(null);
    if (trimmedEmail) {
      try {
        setLoading(true);
        const response = await fetch(formatUrl(window.apiUrls.api_whitelist_email, { email: trimmedEmail }), {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrfCookie()
          },
        });
        if (!response.ok) {
          throw new Error('Failed to add email');
        }
        setEmails(prev => [...prev, trimmedEmail]);
        setNewEmail('');
      } catch (err) {
        console.error(err);
        setError('Error adding email');
      } finally {
        setLoading(false);
      }
    }
  };

  // Delete an email by email
  const handleDelete = async (email: string) => {
    try {
      setLoading(true);
      const response = await fetch(formatUrl(window.apiUrls.api_whitelist_email, { email }), {
        method: 'DELETE',
        headers: {
          'X-CSRFToken': getCsrfCookie()
        },
      });
      if (!response.ok) {
        throw new Error('Failed to delete email');
      }
      setEmails(prev => prev.filter(e => e !== email));
    } catch (err) {
      console.error(err);
      setError('Error deleting email');
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleAddEmail();
    }
  };

  return (
    <div className="max-w-md mx-auto p-4">
      <h2 className="text-xl font-bold text-center mb-4">Whitelisted Emails</h2>
      {error && <div className="text-red-500 mb-4">{error}</div>}
      {loading && <div>Loading...</div>}
      <ul className="list-none p-0">
        {emails.map(email => (
          <li
            key={email}
            className="flex items-center py-2 border-b border-gray-300"
          >
            <span className="flex-grow">{email}</span>
            <button
              onClick={() => handleDelete(email)}
              className="bg-scheme-shade_3 border-none cursor-pointer p-0"
              aria-label={`Delete ${email}`}
            >
              <Trash2 size={18} />
            </button>
          </li>
        ))}
      </ul>
      <div className="mt-4 flex">
        <input
          type="email"
          value={newEmail}
          onChange={(e) => setNewEmail(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Add new email"
          className="flex-grow p-2 bg-scheme-shade_3 text-base border border-gray-300 rounded"
        />
        <button
          onClick={handleAddEmail}
          className="ml-2 p-2 text-base cursor-pointer border border-gray-300 rounded bg-scheme-shade_3"
          disabled={!newEmail.trim() || inputError !== null}
        >
          Add
        </button>
      </div>
      {inputError && <div className="text-red-500 mt-2">{inputError}</div>}
    </div>
  );
};

export default WhitelistEmails;
