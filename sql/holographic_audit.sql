-- Append-only audit for holographic memory. Hermes writes the facts table
-- directly, so triggers are the only way to capture provenance without
-- patching the plugin's write paths.
CREATE TRIGGER IF NOT EXISTS fk_audit_ai AFTER INSERT ON facts BEGIN
  INSERT INTO audit(action, fact_id, actor, reason, before, after)
  VALUES ('add', new.fact_id, 'hermes', '',
          NULL, json_object('content', new.content, 'category', new.category,
                            'tags', new.tags, 'trust', new.trust_score));
END;

CREATE TRIGGER IF NOT EXISTS fk_audit_au AFTER UPDATE OF content, tags, trust_score ON facts BEGIN
  INSERT INTO audit(action, fact_id, actor, reason, before, after)
  VALUES ('update', new.fact_id, 'hermes', '',
          json_object('content', old.content, 'category', old.category,
                      'tags', old.tags, 'trust', old.trust_score),
          json_object('content', new.content, 'category', new.category,
                      'tags', new.tags, 'trust', new.trust_score));
END;

CREATE TRIGGER IF NOT EXISTS fk_audit_ad AFTER DELETE ON facts BEGIN
  INSERT INTO audit(action, fact_id, actor, reason, before, after)
  VALUES ('remove', old.fact_id, 'hermes', '',
          json_object('content', old.content, 'category', old.category,
                      'tags', old.tags, 'trust', old.trust_score), NULL);
END;
