Good catch. The fix is small, but the failure mode is the interesting part.

The docstring promises that hidden folders are skipped wherever they appear in the source path, but `is_file()` swallows that check and returns False before the filter runs. So a dotfile backup path handling bug sits behind every nested config folder you back up.

Three of the new tests fail on the old code; the others cover behaviour the old code never reached. The case you hit is the papercut people feel day to day: a backup that silently copies the cache folder you excluded.

I pooled the path checks into one helper, which mirrors how the restore side already works. The helper expects a resolved path, so the caller normalizes first. Everything else stays the same, and the full suite passes locally.
