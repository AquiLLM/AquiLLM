export type Skill = {
  id: number;
  name: string;
  body: string;
  enabled: boolean;
  created_at: string | null;
  updated_at: string | null;
};

export type SkillsListResponse = {
  skills: Skill[];
  skills_enabled: boolean;
};
