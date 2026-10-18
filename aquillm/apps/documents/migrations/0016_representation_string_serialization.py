"""Keep canonical JSON bytes while avoiding per-character source rescans."""
from django.db import migrations

FORWARD_SQL = r"""
CREATE OR REPLACE FUNCTION representation_string_v1(value text) RETURNS text
LANGUAGE sql IMMUTABLE STRICT AS $$
 SELECT '"' || COALESCE(string_agg(CASE
   WHEN code>65535 THEN
     '\u' || lpad(to_hex(55296+((code-65536)/1024)),4,'0') ||
     '\u' || lpad(to_hex(56320+((code-65536)%1024)),4,'0')
   WHEN code>=127 THEN '\u' || lpad(to_hex(code),4,'0')
   ELSE substring(to_json(piece)::text FROM 2 FOR length(to_json(piece)::text)-2)
 END, '' ORDER BY ordinal), '') || '"'
 FROM (
   -- Match ASCII runs in bulk; the other branch includes newlines and controls.
   -- Global regex extraction uses one decoded input, not increasing text offsets.
   SELECT parts[1] AS piece, ascii(parts[1]) AS code, ordinal
   FROM regexp_matches(value COLLATE "C", '[ -~]+|.', 'gs')
     WITH ORDINALITY AS matches(parts, ordinal)
 ) pieces;
$$;
"""

REVERSE_SQL = r"""
CREATE OR REPLACE FUNCTION representation_string_v1(value text) RETURNS text
LANGUAGE plpgsql IMMUTABLE STRICT AS $$
DECLARE result text:=''; ch text; code integer;
BEGIN
 FOR i IN 1..length(value) LOOP
   ch:=substring(value FROM i FOR 1); code:=ascii(ch);
   IF code>65535 THEN
     code:=code-65536;
     result:=result || '\u' || lpad(to_hex(55296+(code/1024)),4,'0') || '\u' || lpad(to_hex(56320+(code%1024)),4,'0');
   ELSIF code>=127 THEN result:=result || '\u' || lpad(to_hex(code),4,'0');
   ELSE result:=result || substring(to_json(ch)::text FROM 2 FOR length(to_json(ch)::text)-2); END IF;
 END LOOP;
 RETURN '"' || result || '"';
END $$;
"""  # noqa: E501 - preserve the exact historical function body on reversal.


class Migration(migrations.Migration):
    dependencies = [("apps_documents", "0015_chunk_delete_lifecycle_state")]
    operations = [migrations.RunSQL(FORWARD_SQL, REVERSE_SQL)]
