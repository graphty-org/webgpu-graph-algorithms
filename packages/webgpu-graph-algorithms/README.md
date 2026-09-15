# @graphty/webgpu-graph-algorithms

## IMPORTANT NOTICE

**This package is created solely for the purpose of setting up OIDC (OpenID
Connect) trusted publishing with npm.**

This is **NOT** a functional package and contains **NO** code or functionality
beyond the OIDC setup configuration.

## Purpose

This package exists to:

1. Reserve the package name `@graphty/webgpu-graph-algorithms` on the registry
2. Configure OIDC trusted publishing for that name (npm requires the package to
   exist before a trusted publisher can be added)
3. Enable secure, token-less publishing with provenance from CI/CD workflows

## DO NOT USE THIS PACKAGE

This package is a placeholder for OIDC configuration only. It:

- Contains no executable code
- Provides no functionality
- Should not be installed as a dependency
- Exists only for administrative purposes

The real package -- WebGPU-accelerated graph algorithms and layouts over
`@graphty/graph-format` -- will be published under this name from
`graphty-org/graphty-monorepo` once it is ready.

## More Information

- https://docs.npmjs.com/trusted-publishers
- https://docs.npmjs.com/cli/v11/commands/npm-trust
