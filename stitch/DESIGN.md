# Design System Strategy: The Electric Nerve Center

## 1. Overview & Creative North Star
This design system is built for the "Electric Nerve Center"—a high-fidelity digital environment designed for the precision-critical task of smart grid monitoring. Moving away from the generic "admin dashboard" template, this system adopts a **High-End Editorial** aesthetic combined with **Technical Brutalism**. 

The Creative North Star is **"The Luminant Void."** We treat the UI as a dark, expansive space where data is not just displayed, but emitted. By utilizing deep midnight tones as a canvas, we allow the "electric" accents to serve as functional light sources. We break the rigid, symmetrical grid by using intentional white space (breathing room) and asymmetrical data prioritization, ensuring the most critical stress predictors feel authoritative and unavoidable.

---

## 2. Colors: Tonal Depth & Electric Accents
The palette is rooted in the contrast between the infinite dark (`#0a0e14`) and hyper-saturated functional signals.

### The "No-Line" Rule
To achieve a premium, custom feel, **1px solid borders are strictly prohibited** for sectioning or containment. Boundaries must be defined through:
*   **Surface Shifts:** Placing a `surface_container_high` (`#1b2028`) element against a `surface_dim` (`#0a0e14`) background.
*   **Negative Space:** Using the spacing scale to create distinct visual groupings.

### Surface Hierarchy & Nesting
Depth is achieved through physical layering, mimicking stacked sheets of darkened glass:
1.  **Base Layer:** `surface` (#0a0e14) - The foundation of the application.
2.  **Sectional Layer:** `surface_container_low` (#0f141a) - Large areas used to group related modules.
3.  **Component Layer:** `surface_container_highest` (#20262f) - Individual cards or interactive elements.

### The "Glass & Gradient" Rule
For floating elements (modals, tooltips, or high-priority alerts), use **Glassmorphism**. Apply `surface_bright` at 40% opacity with a `20px` backdrop-blur. 
*   **Signature Textures:** Main CTAs should use a subtle linear gradient from `primary` (#81ecff) to `primary_container` (#00e3fd) at a 135-degree angle to provide a "charged" visual energy.

---

## 3. Typography: Technical Authority
We employ a dual-typeface system to balance futuristic technicality with crystalline legibility.

*   **Display & Headlines:** **Space Grotesk.** This is our "Editorial" voice. Use `display-lg` for high-level grid status (e.g., "98% STABLE"). The wide apertures and technical quirks of Space Grotesk convey innovation.
*   **Body & Data Points:** **Inter.** For granular data, Inter provides the reliability required for a stress predictor. Use `label-md` for technical metadata and `title-sm` for card headings.

**Editorial Tip:** Use high-contrast sizing. Pair a massive `display-md` metric with a tiny, uppercase `label-sm` unit to create a sophisticated, data-rich hierarchy.

---

## 4. Elevation & Depth
In the "Luminant Void," depth is conveyed through light and stacking rather than traditional drop shadows.

### The Layering Principle
Achieve a "soft lift" by nesting color tokens. A `surface_container_lowest` card sitting on a `surface_container_low` background creates a natural inset look that feels engineered rather than drawn.

### Ambient Glows (Status Lighting)
Instead of standard shadows, use **Ambient Glows** for status indicators. 
*   **Normal State:** A `secondary` (#2ff801) glow with a 16px blur at 10% opacity.
*   **Critical State:** A `tertiary` (#ff7350) or `error` (#ff716c) glow at 20% opacity. 
The glow color should match the status token to mimic an LED reflecting off a dark surface.

### The "Ghost Border" Fallback
If a container requires a boundary for accessibility, use the **Ghost Border**: the `outline_variant` token (#44484f) at 15% opacity. This defines the edge without interrupting the visual flow of the "void."

---

## 5. Components

### Buttons
*   **Primary:** Gradient fill (`primary` to `primary_container`), `md` (0.375rem) corner radius. Use `on_primary_fixed` for text to maintain high contrast.
*   **Secondary (Outlined):** Ghost Border (15% opacity) with `primary` text. No fill.
*   **States:** On hover, increase the "glow" (shadow-spread) rather than significantly changing the background color.

### Data Cards
*   **Styling:** `surface_container_highest` background. 
*   **Rounding:** `xl` (0.75rem) for the outer container; `md` (0.375rem) for internal nested elements.
*   **Interaction:** No dividers. Use 24px of vertical padding to separate header from content.

### Input Fields
*   **Background:** `surface_container_lowest` (#000000) to create a "recessed" feel.
*   **Typography:** Use Inter `body-md` for user entry. 
*   **Focus State:** The Ghost Border should transition to 100% opacity `primary` (#81ecff) with a subtle `primary` outer glow.

### Status "Pulse" Chips
Unique to this system, use a small 8x8px circle with a CSS animation "pulse" using the `secondary` (Normal) or `tertiary` (Stress) tokens. This adds a sense of "live" data processing.

---

## 6. Do’s and Don’ts

### Do:
*   **Embrace the Dark:** Keep 80% of the UI in the `surface` and `surface_container` range.
*   **Use Asymmetry:** Place the most critical grid metric (e.g., "Stress Index") in a larger, asymmetrical hero card to the left, with smaller supporting metrics to the right.
*   **Prioritize Legibility:** Ensure all neon text on dark backgrounds meets a 4.5:1 contrast ratio.

### Don’t:
*   **Don’t use "Pure" White:** Use `on_surface` (#f1f3fc) or `on_surface_variant` (#a8abb3) for text. Pure white #FFFFFF is too harsh in a dark-mode environment.
*   **Don’t use Dividers:** If you feel the need for a line, increase your padding or shift the background tone by one tier instead.
*   **Don’t over-glow:** Limit the "glow" effect to active status indicators and primary buttons. If everything glows, nothing is important.

### Accessibility Note
While the "Luminant Void" is moody and high-end, data must be readable. Always use the `error` (#ff716c) and `secondary` (#2ff801) tokens for functional feedback, as their high saturation ensures they "pop" against the charcoal background for users with visual impairments.