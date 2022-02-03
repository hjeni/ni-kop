def filter_duplicates_rid(path, parser_class, verbose=False):
    """
    Deletes duplicate rows from a text file based on the row_id (first element on a row)
    """
    # reading all lines to the memory
    raw_f = open(path, "r")
    lines = raw_f.readlines()
    raw_f.close()

    nl_orig = len(lines)
    if verbose:
        print(f'[{path}] Before: {nl_orig}')

    # creating parser instance
    parser = parser_class()

    # writing them down
    rid_prev, cnt = None, 0
    with open(path, 'w') as clean_f:
        for line in lines:
            rid, n, val_ref, res_ref = parser.parse_line(line)
            # skipping duplicates
            if rid_prev != rid:
                clean_f.write(line)
            else:
                cnt += 1
            rid_prev = rid
    if verbose:
        print(f'[{path}] After:  {nl_orig - cnt}')
        print(f'  ~ Change: {cnt}')
