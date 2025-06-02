/// <reference types="@sveltejs/kit" />

declare global {
    namespace App {
        // interface Locals {}
        // interface Platform {}
        // interface Session {}
        // interface Stuff {}
    }

    namespace svelteHTML {
        interface HTMLAttributes<T> {
            [key: string]: any;
        }
    }
}

export {};
