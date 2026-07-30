# jsqr.org

Site built with [Zine](https://zine-ssg.io). Content is in `content/`, layouts in `layouts/`,
static files in `assets/`. Pushing to `main` triggers the GitHub Actions workflow
(`.github/workflows/deploy.yml`), which builds with `zine release` and deploys `public/` to
GitHub Pages.

## Commands

```
zine            # dev server with live reload (http://localhost:1990)
zine --drafts   # include draft pages
zine release    # production build into public/
zine version
```

## Creating a new trail

Trails lie under `content/trails/`, use the `post.shtml` layout, and are listed
automatically on the Trails index via `$page.subpages()`.

1. Create the file:
   - Text only: `content/trails/<slug>.smd`
   - With assets (images, SVGs): `content/trails/<slug>/index.smd`, assets alongside.
2. Add frontmatter:
   ```
   ---
   .title = "Title",
   .date = .date("2026-07-30T00:00:00"),
   .description = "One-line summary",
   .layout = "post.shtml",
   ---
   ```
3. Write the body in SuperMD (`.smd`). See `content/.smd.ziggy-schema`.
4. Preview with `zine`, then commit and push to `main`.

## Adding a project-log entry

The project log is one `.smd` file per year in `content/projects/`.  Each dated entry is a
`## [Title]($section.id("<ISO datetime>"))` section. Entries render in document order, so
add new ones directly below the `## Project snippets` "about" section, newest first.

In `content/projects/<current-year>.smd`, add:

```
## [Entry title]($section.id("2026-07-30T00:00:00"))
Body text for the entry. Links to trails, e.g. [legal
epidemiology](/trails/legal-epidemiology/), are fine here.
```

The `id` timestamp must be unique within the file and is used as the anchor/permalink.
Preview with `zine`, commit, and push.

## Annual project-log maintenance

At the start of a new year, create a fresh year page. The nav "Projects" link and the
Projects Archive update automatically (nav points to the newest year via
`subpages().first?()`; the archive lists all years).

1. Copy the previous year as a template:
   ```
   cp content/projects/2026.smd content/projects/2027.smd
   ```
2. In the new file, update:
   - `.title` and `.date` to the new year (`.date("2027-01-01T00:00:00")`).
   - The "about" paragraph, which names the year ("this page lists entries for 2027").
   - Delete the old year's entries, leaving only the "about" section.
3. Preview with `zine` and confirm the nav "Projects" link points to the new year.
4. Commit and push. Old years remain accessible through the Projects Archive.

## Update the `zine` binary

Zine is installed as a single binary at `~/bin/zine`. The version is also pinned in
two files that must stay in sync with it:

- `zine.ziggy`: `.zine_version = "0.12.0"`
- `.github/workflows/deploy.yml`: `version: v0.12.0` (under "Setup Zine")

To update:

1. Download the new release binary from https://github.com/kristoff-it/zine/releases
   (or the [install docs](https://zine-ssg.io/documentation/)), and replace
   the existing binary. Viz.:
   ```
   mv ~/Downloads/zine ~/bin/zine
   chmod +x ~/bin/zine
   zine version
   ```
2. Bump the version strings in `zine.ziggy` and `.github/workflows/deploy.yml` to match.
3. Run `zine release` locally to confirm the build still succeeds.
4. Commit and push; CI uses the pinned version via `kristoff-it/setup-zine`.
