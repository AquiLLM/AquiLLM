function formatUrl(pattern: string, params: Record<string, string | number>): string {
    return pattern.replace(/%\((\w+)\)s/g, (_, key) => {
      const value = params[key];
      if (value === undefined) {
        throw new Error(`Missing parameter: ${key}`);
      }
      return value.toString();
    });
  }

export default formatUrl;  