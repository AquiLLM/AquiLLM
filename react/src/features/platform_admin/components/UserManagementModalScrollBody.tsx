import React from 'react';
import UserManagementSearchSection from './UserManagementSearchSection';
import UserManagementUsersPanel from './UserManagementUsersPanel';
import type { User, UserWithPermission } from './userManagementTypes';

interface UserManagementModalScrollBodyProps {
  successMessage: string | null;
  error: string | null;
  isLoading: boolean;
  searchQuery: string;
  onSearchQueryChange: (value: string) => void;
  isSearching: boolean;
  searchResults: User[];
  onAddUser: (user: User, permission: 'VIEW' | 'EDIT' | 'MANAGE') => void;
  currentUsers: UserWithPermission[];
  onChangePermission: (userId: number, newPermission: 'VIEW' | 'EDIT' | 'MANAGE') => void;
  onRemoveUser: (userId: number) => void;
}

const UserManagementModalScrollBody: React.FC<UserManagementModalScrollBodyProps> = ({
  successMessage,
  error,
  isLoading,
  searchQuery,
  onSearchQueryChange,
  isSearching,
  searchResults,
  onAddUser,
  currentUsers,
  onChangePermission,
  onRemoveUser,
}) => (
  <div className="overflow-y-auto p-6 flex-grow">
    {successMessage && (
      <div className="mb-4 bg-green-dark text-text-normal p-3 rounded">
        <p>{successMessage}</p>
      </div>
    )}

    {error && (
      <div className="mb-4 bg-red-dark text-text-normal p-3 rounded">
        <p>{error}</p>
      </div>
    )}

    {isLoading ? (
      <div className="flex justify-center items-center py-6">
        <svg
          className="animate-spin h-8 w-8 text-text-normal"
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
    ) : (
      <>
        <UserManagementSearchSection
          searchQuery={searchQuery}
          onSearchQueryChange={onSearchQueryChange}
          isSearching={isSearching}
          searchResults={searchResults}
          onAddUser={onAddUser}
        />
        <UserManagementUsersPanel
          currentUsers={currentUsers}
          onChangePermission={onChangePermission}
          onRemoveUser={onRemoveUser}
        />
      </>
    )}
  </div>
);

export default UserManagementModalScrollBody;
