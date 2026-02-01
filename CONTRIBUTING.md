# Contributing to @browser-ai

First off, huge thank you for considering contributing to this repository! We're incredibly grateful for your interest in making this project even better.

## Issues/Enhancements

If you've encountered an issue/bug/etc., please report it to us.

1. Check the issue tracker before you submit a new issue.
2. Create issue. Please provide a clear title and detailed description.

## Code contributions

Whether you're looking to improve packages, create new example applications or just make minor adjustments, you're more than welcome to do so!
Just make sure to follow the steps below.

### Environment setup

- Node.js (v18 or higher)
- npm

This is a monorepo powered by [turborepo](https://turbo.build/repo/docs) split into the following architecture:

- [`examples/`](./examples/): Example apps showcasing usage of the packages.
- [`packages/`](./packages/): All npm packages in their individual folders.

### Local development

To set up the repository locally, follow these steps:

1. Fork the repository.
2. Clone the repository to your local machine.
3. Install dependencies from the root folder: `npm install`
4. Build the project: `npm run build`. Once built, the `dist` folder is updated and the new code is picked up in the example applications. It is here referenced in the `package.json` with: `"@browser-ai/core": "*"` etc.

You can then run `npm run dev` which will run the local development server and rebuild packages automatically as you make changes.

To test the packages, run `npm run test`.

### Pull requests

To open a PR with changes, do the following:

1. Create new branch. Recommended to name it with alignment to your changes.
2. Commit changes. Make sure your commits are clear and detailed. Preferably follow a [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/#summary) message format.
3. If your changes include an update to the package you were working on, run `npx changeset` to make sure it gets released. Please DON'T use major or minor changesets - only patch.
4. Fix prettier issues. We use prettier to format code files. Please make sure to run `npm run format` and commit any changes.
5. Open the Pull Request. Make sure to link any related issues your PR resolves. Please use a conventional format for the title and include package/example:

- `feat(@browser-ai/core): description`
- `fix(@browser-ai/core): description`
- `chore(examples/next-hybrid): description`

Thank you so much in advance!
