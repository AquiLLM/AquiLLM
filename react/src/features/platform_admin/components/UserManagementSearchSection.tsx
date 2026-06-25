import React from 'react';
import type { User } from './userManagementTypes';

interface UserManagementSearchSectionProps {
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  isSearching: boolean;
  searchResults: User[];
  onAddUser: (user: User, permission: 'VIEW' | 'EDIT' | 'MANAGE') => void;
}

const UserManagementSearchSection: React.FC<UserManagementSearchSectionProps> = ({
  searchQuery,
  onSearchQueryChange,
  isSearching,
  searchResults,
  onAddUser,
}) => (
  <div className="mb-6">
    <label htmlFor="search-users" className="block text-text-low_contrast mb-2">
      Search Users
    </label>
    <div className="relative">
      <input
        id="search-users"
        type="text"
        value={searchQuery}
        onChange={(e) => onSearchQueryChange(e.target.value)}
        placeholder="Type to search users..."
        className="w-full bg-scheme-shade_5 border border-border-mid_contrast rounded px-4 py-2 text-text-normal focus:outline-none focus:ring-2 focus:ring-accent"
      />
      {isSearching && (
        <div className="absolute right-3 top-2">
          <svg
            className="animate-spin h-5 w-5 text-text-normal"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            />
          </svg>
        </div>
      )}
    </div>

    {searchResults.length > 0 && (
      <div className="mt-2 bg-scheme-shade_5 rounded-md overflow-hidden">
        <ul className="divide-y divide-border-mid_contrast">
          {searchResults.map((user) => (
            <li key={user.id} className="p-3 hover:bg-scheme-shade_6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-text-normal">{user.username}</p>
                  <p className="text-text-lower_contrast text-sm">{user.email?.trim() || 'No email'}</p>
                </div>
                <div className="mt-2">
                  <div className="flex space-x-2">
                    <button
                      type="button"
                      className="bg-accent hover:bg-accent-dark text-text-normal px-2 py-1 rounded text-xs"
                      onClick={() => onAddUser(user, 'VIEW')}
                      title="Viewer: Can only view documents - this permission will apply to all subcollections"
                    >
                      Viewer
                    </button>
                    <button
                      type="button"
                      className="bg-green hover:bg-green-dark text-text-normal px-2 py-1 rounded text-xs"
                      onClick={() => onAddUser(user, 'EDIT')}
                      title="Editor: Can add, edit and delete documents - this permission will apply to all subcollections"
                    >
                      Editor
                    </button>
                    <button
                      type="button"
                      className="bg-secondary_accent hover:bg-secondary_accent-dark text-text-normal px-2 py-1 rounded text-xs"
                      onClick={() => onAddUser(user, 'MANAGE')}
                      title="Admin: Can manage users and collection settings - this permission will apply to all subcollections"
                    >
                      Admin
                    </button>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>
    )}
  </div>
);

export default UserManagementSearchSection;
