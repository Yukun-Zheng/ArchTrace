# ArchTrace Web Explorer

The web app is a presentation client for validated ATIR. It never invents or mutates mechanical architecture facts.

## Local development

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:3000`. The app starts with a built-in multimodal policy demo. Drag an ATIR JSON file onto the page, or use **Open ATIR**.

For synchronized source browsing, first create a browser bundle from a repository checkout:

```bash
archtrace web-bundle .archtrace/project.atir.json \
  --source-root /path/to/repository \
  -o .archtrace/project.web.atir.json
```

Source text is stored only under `metadata.web.source_files`; graph nodes, edges, evidence, claims, and conflicts remain unchanged.

## Deployment

This directory is a standard Next.js App Router project. Set the Vercel project root directory to `web`.
