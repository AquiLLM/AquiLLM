/**
 * Dropdown menu component for collection actions
 * @module CollectionSettingsMenu
 */

import React, { useState, useRef, useEffect } from 'react';
import { Collection } from './CollectionsTree';

interface CollectionSettingsMenuProps {
  collection: Collection;                        // Collection to manage
  onMove: (collection: Collection) => void;      // Callback for move action
  onDelete: (collection: Collection) => void;    // Callback for delete action
  onManageCollaborators: (collection: Collection) => void; // Callback for managing collaborators
}

/**
 * Renders a dropdown menu with collection management options
 * @param props - Component properties
 * @param props.collection - Collection being managed
 * @param props.onMove - Function to call when move is selected
 * @param props.onDelete - Function to call when delete is selected
 * @param props.onManageCollaborators - Function to call when manage collaborators is selected
 */
const CollectionSettingsMenu: React.FC<CollectionSettingsMenuProps> = ({
  collection,
  onMove,
  onDelete,
  onManageCollaborators,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  // Handle clicks outside the menu to close it
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div ref={menuRef} style={{ position: 'relative' }}>
      <button
        className='hover:bg-scheme-shade_5 rounded-md p-2'
        style={{ cursor: 'pointer' }}
        onClick={(e) => { e.stopPropagation(); setIsOpen(!isOpen); }}
      >
        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9.99992 12.4999C11.3806 12.4999 12.4999 11.3806 12.4999 9.99992C12.4999 8.61921 11.3806 7.49992 9.99992 7.49992C8.61921 7.49992 7.49992 8.61921 7.49992 9.99992C7.49992 11.3806 8.61921 12.4999 9.99992 12.4999Z" stroke="var(--color-contrast)" stroke-opacity="0.933333" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M16.1666 12.4999C16.0557 12.7513 16.0226 13.0301 16.0716 13.3004C16.1206 13.5707 16.2495 13.8202 16.4416 14.0166L16.4916 14.0666C16.6465 14.2214 16.7695 14.4052 16.8533 14.6075C16.9372 14.8098 16.9804 15.0267 16.9804 15.2458C16.9804 15.4648 16.9372 15.6817 16.8533 15.884C16.7695 16.0863 16.6465 16.2701 16.4916 16.4249C16.3368 16.5799 16.153 16.7028 15.9507 16.7867C15.7483 16.8706 15.5314 16.9137 15.3124 16.9137C15.0934 16.9137 14.8765 16.8706 14.6742 16.7867C14.4719 16.7028 14.288 16.5799 14.1332 16.4249L14.0833 16.3749C13.8869 16.1828 13.6374 16.0539 13.3671 16.0049C13.0967 15.9559 12.8179 15.989 12.5666 16.0999C12.3201 16.2056 12.1099 16.381 11.9618 16.6045C11.8138 16.8281 11.7343 17.0901 11.7333 17.3583V17.4999C11.7333 17.9419 11.5577 18.3659 11.2451 18.6784C10.9325 18.991 10.5086 19.1666 10.0666 19.1666C9.62456 19.1666 9.20063 18.991 8.88807 18.6784C8.57551 18.3659 8.39992 17.9419 8.39992 17.4999V17.4249C8.39347 17.1491 8.30418 16.8816 8.14368 16.6572C7.98317 16.4328 7.75886 16.2618 7.49992 16.1666C7.24857 16.0557 6.96976 16.0226 6.69943 16.0716C6.4291 16.1206 6.17965 16.2495 5.98325 16.4416L5.93325 16.4916C5.77846 16.6465 5.59465 16.7695 5.39232 16.8533C5.18999 16.9372 4.97311 16.9804 4.75408 16.9804C4.53506 16.9804 4.31818 16.9372 4.11585 16.8533C3.91352 16.7695 3.72971 16.6465 3.57492 16.4916C3.41996 16.3368 3.29703 16.153 3.21315 15.9507C3.12928 15.7483 3.08611 15.5314 3.08611 15.3124C3.08611 15.0934 3.12928 14.8765 3.21315 14.6742C3.29703 14.4719 3.41996 14.288 3.57492 14.1332L3.62492 14.0833C3.81703 13.8869 3.94591 13.6374 3.99492 13.3671C4.04394 13.0967 4.01085 12.8179 3.89992 12.5666C3.79428 12.3201 3.61888 12.1099 3.39531 11.9618C3.17173 11.8138 2.90974 11.7343 2.64159 11.7333H2.49992C2.05789 11.7333 1.63397 11.5577 1.32141 11.2451C1.00885 10.9325 0.833252 10.5086 0.833252 10.0666C0.833252 9.62456 1.00885 9.20063 1.32141 8.88807C1.63397 8.57551 2.05789 8.39992 2.49992 8.39992H2.57492C2.85075 8.39347 3.11826 8.30418 3.34267 8.14368C3.56708 7.98317 3.73801 7.75886 3.83325 7.49992C3.94418 7.24857 3.97727 6.96976 3.92826 6.69943C3.87924 6.4291 3.75037 6.17965 3.55825 5.98325L3.50825 5.93325C3.35329 5.77846 3.23036 5.59465 3.14649 5.39232C3.06261 5.18999 3.01944 4.97311 3.01944 4.75408C3.01944 4.53506 3.06261 4.31818 3.14649 4.11585C3.23036 3.91352 3.35329 3.72971 3.50825 3.57492C3.66304 3.41996 3.84685 3.29703 4.04918 3.21315C4.25151 3.12928 4.46839 3.08611 4.68742 3.08611C4.90644 3.08611 5.12332 3.12928 5.32565 3.21315C5.52798 3.29703 5.7118 3.41996 5.86658 3.57492L5.91658 3.62492C6.11298 3.81703 6.36243 3.94591 6.63276 3.99492C6.90309 4.04394 7.1819 4.01085 7.43325 3.89992H7.49992C7.74639 3.79428 7.9566 3.61888 8.10466 3.39531C8.25272 3.17173 8.33218 2.90974 8.33325 2.64159V2.49992C8.33325 2.05789 8.50885 1.63397 8.82141 1.32141C9.13397 1.00885 9.55789 0.833252 9.99992 0.833252C10.4419 0.833252 10.8659 1.00885 11.1784 1.32141C11.491 1.63397 11.6666 2.05789 11.6666 2.49992V2.57492C11.6677 2.84307 11.7471 3.10506 11.8952 3.32864C12.0432 3.55221 12.2534 3.72762 12.4999 3.83325C12.7513 3.94418 13.0301 3.97727 13.3004 3.92826C13.5707 3.87924 13.8202 3.75037 14.0166 3.55825L14.0666 3.50825C14.2214 3.35329 14.4052 3.23036 14.6075 3.14649C14.8098 3.06261 15.0267 3.01944 15.2458 3.01944C15.4648 3.01944 15.6817 3.06261 15.884 3.14649C16.0863 3.23036 16.2701 3.35329 16.4249 3.50825C16.5799 3.66304 16.7028 3.84685 16.7867 4.04918C16.8706 4.25151 16.9137 4.46839 16.9137 4.68742C16.9137 4.90644 16.8706 5.12332 16.7867 5.32565C16.7028 5.52798 16.5799 5.7118 16.4249 5.86658L16.3749 5.91658C16.1828 6.11298 16.0539 6.36243 16.0049 6.63276C15.9559 6.90309 15.989 7.1819 16.0999 7.43325V7.49992C16.2056 7.74639 16.381 7.9566 16.6045 8.10466C16.8281 8.25272 17.0901 8.33218 17.3583 8.33325H17.4999C17.9419 8.33325 18.3659 8.50885 18.6784 8.82141C18.991 9.13397 19.1666 9.55789 19.1666 9.99992C19.1666 10.4419 18.991 10.8659 18.6784 11.1784C18.3659 11.491 17.9419 11.6666 17.4999 11.6666H17.4249C17.1568 11.6677 16.8948 11.7471 16.6712 11.8952C16.4476 12.0432 16.2722 12.2534 16.1666 12.4999Z" stroke="var(--color-contrast)" stroke-opacity="0.933333" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>

      {isOpen && (
        <div 
        className='bg-scheme-shade_3 border border-border-high_contrast'
        style={{
          position: 'absolute',
          right: 0,
          top: '100%',
          borderRadius: '32px',
          boxShadow: '0 0px 10px rgba(0,0,0,0.5)',
          padding: '0.5rem',
          zIndex: 10,
          minWidth: '200px',
        }}>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onManageCollaborators(collection);
              setIsOpen(false);
            }}
            className='text-text-normal'
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-start',
              gap: '0.5rem',
              width: '100%',
              textAlign: 'left',
              padding: '0.5rem 1rem',
              border: 'none',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              borderRadius: '0.25rem',
              textWrap: 'nowrap',
            }}
          >
            
            <svg width="18" height="16" viewBox="0 0 13 11" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M9.20841 10.375V9.29167C9.20841 8.71703 8.98014 8.16593 8.57381 7.7596C8.16748 7.35327 7.61638 7.125 7.04175 7.125H2.70841C2.13378 7.125 1.58268 7.35327 1.17635 7.7596C0.770021 8.16593 0.541748 8.71703 0.541748 9.29167V10.375M12.4584 10.375V9.29167C12.4581 8.8116 12.2983 8.34525 12.0042 7.96584C11.71 7.58642 11.2982 7.31543 10.8334 7.19542M8.66675 0.695417C9.13281 0.814746 9.54589 1.0858 9.84089 1.46583C10.1359 1.84587 10.296 2.31328 10.296 2.79438C10.296 3.27547 10.1359 3.74288 9.84089 4.12292C9.54589 4.50295 9.13281 4.774 8.66675 4.89333M7.04175 2.79167C7.04175 3.98828 6.0717 4.95833 4.87508 4.95833C3.67846 4.95833 2.70841 3.98828 2.70841 2.79167C2.70841 1.59505 3.67846 0.625 4.87508 0.625C6.0717 0.625 7.04175 1.59505 7.04175 2.79167Z" stroke="var(--color-contrast)" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>

            Manage Collaborators
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onMove(collection);
              setIsOpen(false);
            }}
            className='text-text-normal'
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'flex-start',
              gap: '0.5rem',
              width: '100%',
              textAlign: 'left',
              padding: '0.5rem 1rem',
              border: 'none',
              backgroundColor: 'transparent',
              cursor: 'pointer',
              borderRadius: '0.25rem',
            }}
          >

            <svg width="15px" height="15px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 14C22 17.7712 22 19.6569 20.8284 20.8284C19.6569 22 17.7712 22 14 22H10C6.22876 22 4.34315 22 3.17157 20.8284C2.51839 20.1752 2.22937 19.3001 2.10149 18M2 12V6.94975C2 6.06722 2 5.62595 2.06935 5.25839C2.37464 3.64031 3.64031 2.37464 5.25839 2.06935C5.62595 2 6.06722 2 6.94975 2C7.33642 2 7.52976 2 7.71557 2.01738C8.51665 2.09229 9.27652 2.40704 9.89594 2.92051C10.0396 3.03961 10.1763 3.17633 10.4497 3.44975L11 4C11.8158 4.81578 12.2237 5.22367 12.7121 5.49543C12.9804 5.64471 13.2651 5.7626 13.5604 5.84678C14.0979 6 14.6747 6 15.8284 6H16.2021C18.8345 6 20.1506 6 21.0062 6.76946C21.0849 6.84024 21.1598 6.91514 21.2305 6.99383C21.8004 7.62741 21.9482 8.51364 21.9866 10" stroke="var(--color-contrast)" stroke-width="1.5" stroke-linecap="round"/>
              <path d="M2 15C8.44365 15 6.55635 15 13 15M13 15L8.875 12M13 15L8.875 18" stroke="var(--color-contrast)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>

            Move Collection
          </button>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onDelete(collection);
              setIsOpen(false);
            }}
            className="flex items-center justify-start gap-2 w-full text-left py-2 px-4 border-none bg-transparent text-red hover:text-red-dark cursor-pointer rounded"
          >

            <svg width="15" height="17" viewBox="0 0 10 12" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M0.5 3H1.5M1.5 3H9.5M1.5 3V10C1.5 10.2652 1.60536 10.5196 1.79289 10.7071C1.98043 10.8946 2.23478 11 2.5 11H7.5C7.76522 11 8.01957 10.8946 8.20711 10.7071C8.39464 10.5196 8.5 10.2652 8.5 10V3M3 3V2C3 1.73478 3.10536 1.48043 3.29289 1.29289C3.48043 1.10536 3.73478 1 4 1H6C6.26522 1 6.51957 1.10536 6.70711 1.29289C6.89464 1.48043 7 1.73478 7 2V3M4 5.5V8.5M6 5.5V8.5" stroke="currentColor" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>

            Delete
          </button>
        </div>
      )}
    </div>
  );
};

export default CollectionSettingsMenu; 