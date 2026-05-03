"""
reformat_kwargs.py

Rewrites every  kwargs.update( dict( k=v, ... ) )  block in the target
files so that:
  - each key=value pair is on its own line
  - the '=' signs are vertically aligned
  - continuation lines are indented to the column after 'dict( '

Example input:
    kwargs.update( dict( columns = 'x_t', target = 'x_t',
                         lib = [1, 100], pred = [101, 195], E = 3 ) )

Example output:
    kwargs.update( dict( columns = 'x_t',
                         target  = 'x_t',
                         lib     = [1, 100],
                         pred    = [101, 195],
                         E       = 3 ) )
"""

import re
import sys

# ---------------------------------------------------------------------------
# Step 1: extract the full kwargs.update(dict(...)) span from source text,
#         handling nested brackets so list values don't confuse the parser.
# ---------------------------------------------------------------------------

def find_update_blocks( source ):
    """
    Yield (start, end, indent, raw_content) for every kwargs.update( dict(
    block in source, where raw_content is the text between the outer dict(
    and its matching closing ) ).
    """
    pattern = re.compile( r'^(\s*)kwargs\.update\( dict\(', re.MULTILINE )
    for m in pattern.finditer( source ):
        indent  = m.group(1)          # leading whitespace of the line
        # position just after the opening 'dict('
        content_start = m.end()
        depth   = 1
        i       = content_start
        while i < len( source ) and depth > 0:
            if source[i] == '(':
                depth += 1
            elif source[i] == ')':
                depth -= 1
            i += 1
        # i now points to the character after the dict(...) closing ')'.
        # Advance past optional whitespace and the outer ')' of .update().
        j = i
        while j < len( source ) and source[j] in (' ', '\t'):
            j += 1
        if j < len( source ) and source[j] == ')':
            j += 1   # consume the closing ')' of .update()
        yield m.start(), j, indent, source[ content_start : i - 1 ]


# ---------------------------------------------------------------------------
# Step 2: parse key = value pairs from the raw content string.
#         Values can contain nested brackets, quotes, etc.
# ---------------------------------------------------------------------------

def parse_pairs( content ):
    """
    Return list of (key, value) strings from a comma-separated
    'k = v, k = v, ...' string.  Respects nested brackets and quotes.
    """
    pairs  = []
    depth  = 0
    in_str = None
    token  = []
    i      = 0

    while i < len( content ):
        ch = content[i]

        # track string literals so commas inside strings are ignored
        if in_str:
            token.append( ch )
            if ch == in_str and ( i == 0 or content[i-1] != '\\' ):
                in_str = None
            i += 1
            continue

        if ch in ('"', "'"):
            # detect triple-quotes
            triple = content[i:i+3]
            if triple in ('"""', "'''"):
                in_str = triple
                token.append( triple )
                i += 3
                continue
            in_str = ch
            token.append( ch )
            i += 1
            continue

        if ch in ('(', '[', '{'):
            depth += 1
            token.append( ch )
        elif ch in (')', ']', '}'):
            depth -= 1
            token.append( ch )
        elif ch == ',' and depth == 0:
            raw = ''.join( token ).strip()
            if raw:
                pairs.append( raw )
            token = []
            i += 1
            continue
        else:
            token.append( ch )
        i += 1

    raw = ''.join( token ).strip()
    if raw:
        pairs.append( raw )

    # split each 'key = value' token on the first '='
    result = []
    for pair in pairs:
        if '=' in pair:
            key, _, val = pair.partition( '=' )
            result.append( ( key.strip(), val.strip() ) )
    return result


# ---------------------------------------------------------------------------
# Step 3: format aligned replacement text.
# ---------------------------------------------------------------------------

def format_block( indent, pairs ):
    """
    Return the replacement kwargs.update( dict( ... ) ) string with
    one key = value per line and '=' signs column-aligned.
    """
    if not pairs:
        return f"{indent}kwargs.update( dict() )"

    max_key = max( len(k) for k, _ in pairs )
    prefix  = f"{indent}kwargs.update( dict( "
    cont    = ' ' * len( prefix )          # continuation indent

    lines = []
    for idx, (key, val) in enumerate( pairs ):
        padding = ' ' * ( max_key - len(key) )
        is_last = ( idx == len(pairs) - 1 )
        comma   = '' if is_last else ','
        leader  = prefix if idx == 0 else cont
        if is_last:
            lines.append( f"{leader}{key}{padding} = {val} ) )" )
        else:
            lines.append( f"{leader}{key}{padding} = {val}{comma}" )

    return '\n'.join( lines )


# ---------------------------------------------------------------------------
# Step 4: apply to a file.
# ---------------------------------------------------------------------------

def reformat_file( path ):
    with open( path, 'r' ) as fh:
        source = fh.read()

    blocks = list( find_update_blocks( source ) )
    if not blocks:
        return

    # Process in reverse order so offsets stay valid as we replace.
    result = source
    for start, end, indent, content in reversed( blocks ):
        pairs       = parse_pairs( content )
        replacement = format_block( indent, pairs )
        result      = result[:start] + replacement + result[end:]

    with open( path, 'w' ) as fh:
        fh.write( result )
    print( f"Reformatted: {path}  ({len(blocks)} block(s))" )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for path in sys.argv[1:]:
        reformat_file( path )
