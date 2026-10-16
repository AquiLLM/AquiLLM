"""Pure lexical transition shared by scalar and batched copy directory seeks."""


def directory_step(record, name, low, high):
    from .frontier_reads_v2 import schema_error

    kind, node = record
    if kind == "directory" and type(node) is dict and "layout" in node:
        from .frontier_packed_directory_v3 import packed_step

        try:
            return packed_step(node, name, low, high)
        except ValueError as error:
            raise schema_error() from error
    if (
        kind != "directory"
        or type(node) is not dict
        or set(node) != {"name", "head", "left", "right"}
    ):
        raise schema_error()
    current = node["name"]
    if type(current) is not str or not low < current < high:
        raise schema_error()
    if current == name:
        return True, node["head"], low, high
    if name < current:
        return False, node["left"], low, current
    return False, node["right"], current, high
