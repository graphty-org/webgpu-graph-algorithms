# WebGPU Graph Algorithms

A high-performance graph algorithms library leveraging WebGPU for parallel computation in the browser.

## Features

- GPU-accelerated graph algorithms using WebGPU
- TypeScript-first with full type safety
- Zero dependencies for core functionality
- Comprehensive test suite

## Requirements

**WebGPU is mandatory** - this library has no CPU fallbacks. You must use:
- Chrome/Edge 113+ 
- Safari 18+ (Technology Preview)
- Firefox 128+ (with WebGPU enabled)

Tests will fail if WebGPU is not available.

## Getting Started

### Prerequisites

- Node.js 18 or higher
- A browser with WebGPU support (Chrome 113+, Edge 113+, Safari 18+, Firefox 128+)

### Installation

```bash
npm install
```

### Development

1. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your settings
```

2. Run development server:
```bash
# Run development server (uses settings from .env)
npm run dev

# Override environment variables
HOST=localhost PORT=9015 npm run dev

# Run all tests
npm test

# Run tests once (no watch mode)
npm run test:run

# Run WebGPU availability check
npm run test:webgpu

# Run tests with UI
npm run test:ui

# Run tests with coverage
npm run test:coverage

# Run benchmarks
npm run benchmark

# Type checking
npm run typecheck

# Linting
npm run lint

# Build for production
npm run build
```

## Project Structure

```
src/
├── algorithms/     # Graph algorithm implementations
├── core/          # WebGPU setup and utilities
├── formats/       # Graph format converters
├── types/         # TypeScript type definitions
└── index.ts       # Main entry point

test/
├── algorithms/    # Algorithm tests
├── core/         # Core functionality tests
└── helpers/      # Test utilities
```

## Browser Compatibility

This library requires WebGPU support:
- Chrome/Edge 113+
- Safari 18+ (Technology Preview)
- Firefox 128+ (behind flag)

## License

MIT