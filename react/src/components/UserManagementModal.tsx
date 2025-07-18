import React, { useState, useEffect, useRef } from 'react';
import { Collection } from './CollectionsTree';
import { getCookie } from '../utils/csrf';
import formatUrl from '../utils/formatUrl';
// Define the types for users and permissions
interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
}

interface CollectionPermissions {
  viewers: number[];
  editors: number[];
  admins: number[];
}

interface UserWithPermission extends User {
  permission: 'VIEW' | 'EDIT' | 'MANAGE';
}

interface UserManagementModalProps {
  collection: Collection | null;
  isOpen: boolean;
  onClose: () => void;
  onSave: () => void;
}

const UserManagementModal: React.FC<UserManagementModalProps> = ({
  collection,
  isOpen,
  onClose,
  onSave,
}) => {
  // State for the modal
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchResults, setSearchResults] = useState<User[]>([]);
  const [isSearching, setIsSearching] = useState<boolean>(false);
  const [currentUsers, setCurrentUsers] = useState<UserWithPermission[]>([]);
  const [permissions, setPermissions] = useState<CollectionPermissions>({
    viewers: [],
    editors: [],
    admins: [],
  });
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear states when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setError(null);
      setSuccessMessage(null);
    }
  }, [isOpen]);

  // Fetch current users with permissions when the modal is opened
  useEffect(() => {
    if (isOpen && collection) {
      fetchCurrentUsers();
    }
  }, [isOpen, collection]);

  // Search for users when the search query changes
  useEffect(() => {
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (searchQuery.trim().length > 0) {
      searchTimeoutRef.current = setTimeout(() => {
        searchUsers(searchQuery);
      }, 300); // Debounce search for 300ms
    } else {
      setSearchResults([]);
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [searchQuery]);

  const fetchCurrentUsers = async () => {
    if (!collection) return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(formatUrl(window.apiUrls.api_collection_permissions, { col_id: collection.id }), {
        headers: {
          'Accept': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch current users');
      }

      const data = await response.json();
      
      // Transform the data to our internal format
      const viewers = data.viewers || [];
      const editors = data.editors || [];
      const admins = data.admins || [];

      setPermissions({
        viewers: viewers.map((user: User) => user.id),
        editors: editors.map((user: User) => user.id),
        admins: admins.map((user: User) => user.id),
      });

      // Combine all users with their permissions
      const allUsers: UserWithPermission[] = [
        ...viewers.map((user: User) => ({ ...user, permission: 'VIEW' as const })),
        ...editors.map((user: User) => ({ ...user, permission: 'EDIT' as const })),
        ...admins.map((user: User) => ({ ...user, permission: 'MANAGE' as const })),
      ];

      setCurrentUsers(allUsers);
    } catch (error) {
      console.error('Error fetching users:', error);
      setError(error instanceof Error ? error.message : 'Failed to fetch current users');
    } finally {
      setIsLoading(false);
    }
  };

  const searchUsers = async (query: string) => {
    setIsSearching(true);
    try {
      const response = await fetch(window.apiUrls.api_search_users + `?query=${encodeURIComponent(query)}&exclude_current=true`, {
        headers: {
          'Accept': 'application/json',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        throw new Error('Failed to search users');
      }

      const data = await response.json();
      setSearchResults(data.users || []);
    } catch (error) {
      console.error('Error searching users:', error);
      setError('Failed to search users');
    } finally {
      setIsSearching(false);
    }
  };

  const handleSave = async () => {
    if (!collection) return;
    
    setIsSaving(true);
    setError(null);
    setSuccessMessage(null);

    try {
      const response = await fetch(formatUrl(window.apiUrls.api_collection_permissions, { col_id: collection.id }), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-CSRFToken': getCookie('csrftoken'),
        },
        body: JSON.stringify(permissions),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        let errorMessage = 'Failed to update permissions';
        if (errorData.error && errorData.error.includes('User matching query does not exist')) {
          errorMessage = 'One or more users could not be found, possibly because they were deleted. Please close and reopen the dialog to refresh the list, or try removing the affected users.';
        } else {
          errorMessage = errorData.error || 'Failed to update permissions';
        }
        throw new Error(errorMessage);
      }

      setSuccessMessage('Permissions updated successfully!');
      
      // After a short delay, close the modal and notify parent component
      setTimeout(() => {
        onSave();
        onClose();
      }, 1500);
    } catch (error) {
      console.error('Error saving permissions:', error);
      let errorMessage = 'Failed to update permissions';
      if (error instanceof Error) {
        if (error.message.includes('User matching query does not exist')) {
          errorMessage = 'One or more users could not be found, possibly because they were deleted. Please close and reopen the dialog to refresh the list, or try removing the affected users.';
        } else {
          errorMessage = error.message;
        }
      }
      setError(errorMessage);
    } finally {
      setIsSaving(false);
    }
  };

  const addUser = (user: User, permission: 'VIEW' | 'EDIT' | 'MANAGE') => {
    // First, ensure the user is not already in any permission group
    const newPermissions = {
      viewers: permissions.viewers.filter(id => id !== user.id),
      editors: permissions.editors.filter(id => id !== user.id),
      admins: permissions.admins.filter(id => id !== user.id),
    };

    // Add the user to the appropriate permission group
    if (permission === 'VIEW') {
      newPermissions.viewers.push(user.id);
    } else if (permission === 'EDIT') {
      newPermissions.editors.push(user.id);
    } else if (permission === 'MANAGE') {
      newPermissions.admins.push(user.id);
    }

    setPermissions(newPermissions);

    // Update the current users list
    const existingUserIndex = currentUsers.findIndex(u => u.id === user.id);
    if (existingUserIndex >= 0) {
      const updatedUsers = [...currentUsers];
      updatedUsers[existingUserIndex] = { ...updatedUsers[existingUserIndex], permission };
      setCurrentUsers(updatedUsers);
    } else {
      setCurrentUsers([...currentUsers, { ...user, permission }]);
    }

    // Clear search results and query
    setSearchResults([]);
    setSearchQuery('');
  };

  const removeUser = (userId: number) => {
    setPermissions({
      viewers: permissions.viewers.filter(id => id !== userId),
      editors: permissions.editors.filter(id => id !== userId),
      admins: permissions.admins.filter(id => id !== userId),
    });

    setCurrentUsers(currentUsers.filter(user => user.id !== userId));
  };

  const changeUserPermission = (userId: number, newPermission: 'VIEW' | 'EDIT' | 'MANAGE') => {
    // Remove from all permission groups
    const newPermissions = {
      viewers: permissions.viewers.filter(id => id !== userId),
      editors: permissions.editors.filter(id => id !== userId),
      admins: permissions.admins.filter(id => id !== userId),
    };

    // Add to the appropriate group
    if (newPermission === 'VIEW') {
      newPermissions.viewers.push(userId);
    } else if (newPermission === 'EDIT') {
      newPermissions.editors.push(userId);
    } else if (newPermission === 'MANAGE') {
      newPermissions.admins.push(userId);
    }

    setPermissions(newPermissions);

    // Update the user's permission in the current users list
    const updatedUsers = currentUsers.map(user => 
      user.id === userId ? { ...user, permission: newPermission } : user
    );
    setCurrentUsers(updatedUsers);
  };

  if (!isOpen || !collection) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-scheme-shade_3 rounded-lg shadow-xl w-full max-w-2xl max-h-[90vh] overflow-hidden flex flex-col">
        <div className="px-6 py-4 border-b border-border-low_contrast flex justify-between items-center">
          <h2 className="text-xl font-semibold">
            Manage Collaborators: {collection?.name}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Help information about permission inheritance */}
        <div className="px-6 py-3 bg-scheme-shade_4 border-b border-border-lower_contrast">
          <div className="flex items-start">
            <svg className="w-5 h-5 mt-0.5 mr-2 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h3 className="text-sm font-semibold text-blue-400 mb-1">About Permissions</h3>
              <p className="text-xs text-gray-300 mb-2">
                Users with permissions on a collection automatically inherit access to all subcollections:
              </p>
              <ul className="text-xs text-gray-300 list-disc list-inside ml-2 space-y-1">
                <li><span className="font-semibold">Viewer</span> - Can view documents but cannot make changes</li>
                <li><span className="font-semibold">Editor</span> - Can add, edit, and delete documents</li>
                <li><span className="font-semibold">Admin</span> - Can manage users and collection settings</li>
              </ul>
            </div>
          </div>
        </div>

        <div className="overflow-y-auto p-6 flex-grow">
          {/* Success message */}
          {successMessage && (
            <div className="mb-4 bg-green-900 text-white p-3 rounded">
              <p>{successMessage}</p>
            </div>
          )}

          {/* Error message */}
          {error && (
            <div className="mb-4 bg-red-900 text-white p-3 rounded">
              <p>{error}</p>
            </div>
          )}

          {/* Loading indicator */}
          {isLoading ? (
            <div className="flex justify-center items-center py-6">
              <svg className="animate-spin h-8 w-8 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
            </div>
          ) : (
            <>
              <div className="mb-6">
                <label htmlFor="search-users" className="block text-gray-300 mb-2">
                  Search Users
                </label>
                <div className="relative">
                  <input
                    id="search-users"
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Type to search users..."
                    className="w-full bg-gray-700 border border-gray-600 rounded px-4 py-2 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                  {isSearching && (
                    <div className="absolute right-3 top-2">
                      <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </div>
                  )}
                </div>

                {searchResults.length > 0 && (
                  <div className="mt-2 bg-gray-700 rounded-md overflow-hidden">
                    <ul className="divide-y divide-gray-600">
                      {searchResults.map((user) => (
                        <li key={user.id} className="p-3 hover:bg-gray-600">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="text-white">{user.username}</p>
                              <p className="text-gray-400 text-sm">{user.email?.trim() || 'No email'}</p>
                            </div>
                            <div className="mt-2">
                              <div className="flex space-x-2">
                                <button
                                  className="bg-blue-600 hover:bg-blue-700 text-white px-2 py-1 rounded text-xs"
                                  onClick={() => addUser(user, 'VIEW')}
                                  title="Viewer: Can only view documents - this permission will apply to all subcollections"
                                >
                                  Viewer
                                </button>
                                <button
                                  className="bg-green-600 hover:bg-green-700 text-white px-2 py-1 rounded text-xs"
                                  onClick={() => addUser(user, 'EDIT')}
                                  title="Editor: Can add, edit and delete documents - this permission will apply to all subcollections"
                                >
                                  Editor
                                </button>
                                <button
                                  className="bg-purple-600 hover:bg-purple-700 text-white px-2 py-1 rounded text-xs"
                                  onClick={() => addUser(user, 'MANAGE')}
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

              <div>
                <h3 className="text-white text-lg font-medium mb-2">Current Collaborators</h3>
                {currentUsers.length === 0 ? (
                  <p className="text-gray-400">No collaborators yet.</p>
                ) : (
                  <ul className="divide-y divide-gray-700 bg-gray-800 rounded-md overflow-hidden">
                    {currentUsers.map((user) => (
                      <li key={user.id} className="p-3">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-white">{user.username}</p>
                            <p className="text-gray-400 text-sm">{user.email?.trim() || 'No email'}</p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <button
                              className={`px-2 py-1 rounded text-xs ${
                                user.permission === 'VIEW'
                                  ? 'bg-blue-600 text-white'
                                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                              }`}
                              onClick={() => changeUserPermission(user.id, 'VIEW')}
                              title="Viewer: Can only view documents - this permission will apply to all subcollections"
                            >
                              Viewer
                            </button>
                            <button
                              className={`px-2 py-1 rounded text-xs ${
                                user.permission === 'EDIT'
                                  ? 'bg-green-600 text-white'
                                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                              }`}
                              onClick={() => changeUserPermission(user.id, 'EDIT')}
                              title="Editor: Can add, edit and delete documents - this permission will apply to all subcollections"
                            >
                              Editor
                            </button>
                            <button
                              className={`px-2 py-1 rounded text-xs ${
                                user.permission === 'MANAGE'
                                  ? 'bg-purple-600 text-white'
                                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                              }`}
                              onClick={() => changeUserPermission(user.id, 'MANAGE')}
                              title="Admin: Can manage users and collection settings - this permission will apply to all subcollections"
                            >
                              Admin
                            </button>
                            <button
                              onClick={() => removeUser(user.id)}
                              className="text-red-500 hover:text-red-400"
                            >
                              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                              </svg>
                            </button>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </>
          )}
        </div>

        <div className="border-t border-gray-700 px-6 py-4 flex justify-end space-x-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-white bg-gray-600 hover:bg-gray-500 rounded focus:outline-none"
            disabled={isSaving}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            disabled={isSaving || isLoading}
            className="px-4 py-2 text-white bg-blue-600 hover:bg-blue-500 rounded focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default UserManagementModal;
