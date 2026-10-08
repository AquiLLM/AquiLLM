import type { ContextCollection, ContextSkill, SkillOverrides } from '../types';

export const skillEnabled = (skill: ContextSkill, selected: Set<string>, overrides: SkillOverrides) =>
  Object.prototype.hasOwnProperty.call(overrides, skill.id)
    ? overrides[skill.id]
    : skill.default_collection_ids.some(id => selected.has(String(id)));

export function descendantIds(collections: ContextCollection[], id: string): string[] {
  const found = new Set([id]);
  const queue = [id];
  for (let index = 0; index < queue.length; index++) {
    for (const collection of collections) {
      const child = String(collection.id);
      if (collection.parent != null && String(collection.parent) === queue[index] && !found.has(child)) {
        found.add(child);
        queue.push(child);
      }
    }
  }
  return queue;
}

export function validOverrides(skills: ContextSkill[], overrides: SkillOverrides): SkillOverrides {
  return Object.fromEntries(skills.filter(skill => Object.prototype.hasOwnProperty.call(overrides, skill.id))
    .map(skill => [skill.id, overrides[skill.id]]));
}

export const payloadCollectionId = (id: string): string | number => {
  const numeric = Number(id);
  return Number.isInteger(numeric) && String(numeric) === id ? numeric : id;
};
