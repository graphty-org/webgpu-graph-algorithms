import js from "@eslint/js";
import stylistic from "@stylistic/eslint-plugin";
import simpleImportSort from "eslint-plugin-simple-import-sort";
import tseslint from "typescript-eslint";
import globals from "globals";

export default tseslint.config(
    js.configs.recommended,
    ...tseslint.configs.strictTypeChecked,
    ...tseslint.configs.stylisticTypeChecked,
    {
        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.node
            },
            parserOptions: {
                projectService: true,
                tsconfigRootDir: import.meta.dirname
            }
        }
    },
    {
        plugins: {
            "@stylistic": stylistic,
            "simple-import-sort": simpleImportSort
        },
        rules: {
            // Import sorting
            "simple-import-sort/imports": "error",
            "simple-import-sort/exports": "error",
            
            // Code style
            "@stylistic/indent": ["error", 4],
            "@stylistic/quotes": ["error", "double"],
            "@stylistic/semi": ["error", "always"],
            "@stylistic/member-delimiter-style": "error",
            "@stylistic/comma-dangle": ["error", "never"],
            "@stylistic/array-bracket-spacing": ["error", "never"],
            "@stylistic/object-curly-spacing": ["error", "always"],
            "@stylistic/space-before-function-paren": ["error", {
                anonymous: "always",
                named: "never",
                asyncArrow: "always"
            }],
            "@stylistic/keyword-spacing": "error",
            "@stylistic/space-infix-ops": "error",
            "@stylistic/no-trailing-spaces": "error",
            "@stylistic/eol-last": ["error", "always"],
            "@stylistic/no-multiple-empty-lines": ["error", { max: 1 }],
            
            // TypeScript specific
            "@typescript-eslint/explicit-function-return-type": ["error", {
                allowExpressions: true,
                allowTypedFunctionExpressions: true
            }],
            "@typescript-eslint/naming-convention": [
                "error",
                {
                    selector: "default",
                    format: ["camelCase"],
                    leadingUnderscore: "allow",
                    trailingUnderscore: "allow"
                },
                {
                    selector: "import",
                    format: ["camelCase", "PascalCase"]
                },
                {
                    selector: "variable",
                    format: ["camelCase", "UPPER_CASE", "PascalCase"],
                    leadingUnderscore: "allow",
                    trailingUnderscore: "allow"
                },
                {
                    selector: "typeLike",
                    format: ["PascalCase"]
                },
                {
                    selector: "enumMember",
                    format: ["PascalCase", "UPPER_CASE"]
                },
                {
                    selector: "property",
                    format: ["camelCase", "PascalCase", "UPPER_CASE"],
                    leadingUnderscore: "allow"
                },
                {
                    selector: "method",
                    format: ["camelCase"],
                    leadingUnderscore: "allow"
                }
            ],
            
            // Best practices
            "eqeqeq": ["error", "always"],
            "no-console": ["warn", { allow: ["warn", "error"] }],
            "no-debugger": "error",
            "no-eval": "error",
            "no-implied-eval": "error",
            "prefer-const": "error",
            "no-var": "error",
            "object-shorthand": "error",
            "prefer-template": "error",
            
            // Disabled rules
            "@typescript-eslint/no-unused-vars": ["error", {
                argsIgnorePattern: "^_",
                varsIgnorePattern: "^_"
            }],
            "@typescript-eslint/restrict-template-expressions": "off",
            "@typescript-eslint/no-non-null-assertion": "off"
        }
    },
    {
        files: ["**/*.js", "**/*.mjs"],
        ...tseslint.configs.disableTypeChecked
    },
    {
        ignores: [
            "**/dist/**",
            "**/coverage/**",
            "**/node_modules/**",
            "**/*.d.ts"
        ]
    }
);