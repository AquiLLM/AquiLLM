import React from 'react';
import type { UserWithPermission } from './userManagementTypes';

interface UserManagementUsersPanelProps {
  currentUsers: UserWithPermission[];
  onChangePermission: (userId: number, newPermission: 'VIEW' | 'EDIT' | 'MANAGE') => void;
  onRemoveUser: (userId: number) => void;
}

const UserManagementUsersPanel: React.FC<UserManagementUsersPanelProps> = ({
  currentUsers,
  onChangePermission,
  onRemoveUser,
}) => (
  <div>
    <h3 className="text-text-normal text-lg font-medium mb-2">Current Collaborators</h3>
    {currentUsers.length === 0 ? (
      <p className="text-text-lower_contrast">No collaborators yet.</p>
    ) : (
      <ul className="divide-y divide-border-low_contrast bg-scheme-shade_4 rounded-md overflow-hidden">
        {currentUsers.map((user) => (
          <li key={user.id} className="p-3">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-text-normal">{user.username}</p>
                <p className="text-text-lower_contrast text-sm">{user.email?.trim() || 'No email'}</p>
              </div>
              <div className="flex items-center space-x-2">
                <button
                  type="button"
                  className={`px-2 py-1 rounded text-xs ${
                    user.permission === 'VIEW'
                      ? 'bg-accent text-text-normal'
                      : 'bg-scheme-shade_5 text-text-low_contrast hover:bg-scheme-shade_6'
                  }`}
                  onClick={() => onChangePermission(user.id, 'VIEW')}
                  title="Viewer: Can only view documents - this permission will apply to all subcollections"
                >
                  Viewer
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded text-xs ${
                    user.permission === 'EDIT'
                      ? 'bg-green text-text-normal'
                      : 'bg-scheme-shade_5 text-text-low_contrast hover:bg-scheme-shade_6'
                  }`}
                  onClick={() => onChangePermission(user.id, 'EDIT')}
                  title="Editor: Can add, edit and delete documents - this permission will apply to all subcollections"
                >
                  Editor
                </button>
                <button
                  type="button"
                  className={`px-2 py-1 rounded text-xs ${
                    user.permission === 'MANAGE'
                      ? 'bg-secondary_accent text-text-normal'
                      : 'bg-scheme-shade_5 text-text-low_contrast hover:bg-scheme-shade_6'
                  }`}
                  onClick={() => onChangePermission(user.id, 'MANAGE')}
                  title="Admin: Can manage users and collection settings - this permission will apply to all subcollections"
                >
                  Admin
                </button>
                <button
                  type="button"
                  onClick={() => onRemoveUser(user.id)}
                  className="text-red hover:text-red-dark"
                >
                  <svg
                    className="w-5 h-5"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </li>
        ))}
      </ul>
    )}
  </div>
);

export default UserManagementUsersPanel;
