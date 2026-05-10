---
name: NeuroGlass Medical AI
colors:
  surface: '#f3fbfa'
  surface-dim: '#d4dbdb'
  surface-bright: '#f3fbfa'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eef5f5'
  surface-container: '#e8efef'
  surface-container-high: '#e2eae9'
  surface-container-highest: '#dce4e3'
  on-surface: '#161d1d'
  on-surface-variant: '#3b494a'
  inverse-surface: '#2a3232'
  inverse-on-surface: '#ebf2f2'
  outline: '#6b7a7a'
  outline-variant: '#bac9c9'
  surface-tint: '#00696b'
  primary: '#00696b'
  on-primary: '#ffffff'
  primary-container: '#2de2e6'
  on-primary-container: '#006163'
  inverse-primary: '#1edce0'
  secondary: '#4d5f7d'
  on-secondary: '#ffffff'
  secondary-container: '#c8dbfe'
  on-secondary-container: '#4e607e'
  tertiary: '#5b5f61'
  on-tertiary: '#ffffff'
  tertiary-container: '#cacdd0'
  on-tertiary-container: '#54575a'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#52f9fc'
  primary-fixed-dim: '#1edce0'
  on-primary-fixed: '#002021'
  on-primary-fixed-variant: '#004f51'
  secondary-fixed: '#d6e3ff'
  secondary-fixed-dim: '#b5c7ea'
  on-secondary-fixed: '#071c36'
  on-secondary-fixed-variant: '#364764'
  tertiary-fixed: '#e0e3e6'
  tertiary-fixed-dim: '#c4c7ca'
  on-tertiary-fixed: '#191c1e'
  on-tertiary-fixed-variant: '#44474a'
  background: '#f3fbfa'
  on-background: '#161d1d'
  surface-variant: '#dce4e3'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-md:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
  body-lg:
    fontFamily: Inter
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-caps:
    fontFamily: Space Grotesk
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.1em
  data-mono:
    fontFamily: Space Grotesk
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  unit: 4px
  gutter: 24px
  margin-desktop: 40px
  panel-padding: 20px
  stack-sm: 8px
  stack-md: 16px
  stack-lg: 32px
---

## Brand & Style

The design system is engineered for high-stakes clinical environments where cognitive load management is paramount. It bridges the gap between futuristic data visualization and rigorous medical utility. The aesthetic is defined by **Glassmorphism**, utilizing multi-layered translucency to suggest depth and focus without cluttering the visual field. 

The brand personality is authoritative, surgical, and hyper-modern. It evokes a sense of "digital clarity"—where AI insights are presented as ethereal yet precise overlays on a grounded, stable infrastructure. The target audience includes radiologists, neurosurgeons, and diagnostic technicians who require a "cockpit" experience that feels both high-fidelity and calming during critical decision-making.

## Colors

This design system utilizes a high-contrast clinical palette optimized for long-shift legibility. 

- **Medical Cyan (#2DE2E6)** is the primary action and data-highlight color, representing active AI scanning and neural pathways.
- **Deep Navy (#0B1F3A)** is used for high-level information headers and primary text to provide a grounding weight.
- **Background Off-white (#F7F9FC)** acts as the canvas, providing a sanitized, professional environment that minimizes eye strain compared to pure white.
- **Amber Warning** and **Soft Red** are reserved strictly for diagnostic alerts and critical system failures, ensuring immediate visual redirection.

## Typography

The typographic system prioritizes functional hierarchy. **Inter** is the primary workhorse, chosen for its exceptional legibility in dense medical interfaces. For technical data points and "cockpit" labels, **Space Grotesk** is introduced to provide a subtle technological edge and better character differentiation for numerical strings.

- **Headlines:** Use tight tracking and semi-bold weights to anchor panels.
- **Data Labels:** Always in uppercase Space Grotesk to distinguish metadata from patient narrative text.
- **Micro-copy:** Maintain a minimum size of 12px to ensure accessibility in low-light diagnostic rooms.

## Layout & Spacing

The layout utilizes a **Fixed Grid** philosophy for the primary diagnostic tools to ensure muscle memory for clinicians, while the lateral information panels use a **Fluid** model.

- **The Cockpit Grid:** A 12-column system with wide 24px gutters to allow the glassmorphism shadows room to "breathe" without overlapping adjacent content.
- **Z-Axis Strategy:** Essential diagnostic images (MRI/CT) reside on the base layer. AI insights and floating cards occupy the +1 and +2 elevation layers.
- **Mobile/Tablet:** For bedside rounds, the 12-column grid collapses to a 4-column stack, prioritizing the "Radial Gauges" and "Action Badges" at the top of the viewport.

## Elevation & Depth

Depth is the primary navigator in this design system. It is achieved through:

1.  **Backdrop Blurs:** Every floating panel must have a `backdrop-filter: blur(20px)` and a background opacity of 60-80% using the Background Off-white.
2.  **Inner Glows:** To simulate "Medical Glass," panels feature a 1px internal stroke of Medical Cyan at 20% opacity on the top and left edges.
3.  **Soft Ambient Shadows:** Instead of black shadows, use a diffused Deep Navy shadow (#0B1F3A at 8% opacity) with a 30px blur to create a lifting effect without heavy "dirtiness" on the UI.
4.  **Tonal Tiers:** The further back an element is in the hierarchy, the higher its opacity and lower its blur intensity.

## Shapes

The shape language is "Softly Technical." We avoid aggressive sharp corners to maintain a professional, approachable medical feel, but avoid full pill-shapes for primary containers to keep the UI looking "industrial."

- **Panels & Cards:** 16px (rounded-lg) provides a modern, high-end hardware feel.
- **Interactive Elements:** Buttons and inputs use an 8px (soft) radius to signal clickability.
- **Medical Badges:** Use a 4px radius for a "tag" appearance that fits within tight data rows.
- **Gauges:** Use perfect circles with open-ended paths to visualize neural probability and diagnostic confidence.

## Components

### Floating Glass Cards
The signature component. Use a semi-transparent white fill with a thin cyan border. Content should be grouped with 16px internal padding. The header of the card should use the `label-caps` typography style.

### Radial Gauges
Used for "AI Confidence" and "Vital Probability." These are 3/4 rings in Medical Cyan. The center of the gauge should display the percentage in `headline-md` weight. When the value drops below a threshold, the stroke color transitions to Amber Warning.

### Medical Badges
Small, high-contrast indicators. For example, a "CRITICAL" badge uses a Soft Red background with white text, while a "STABLE" badge uses a Deep Navy background with Medical Cyan text.

### Buttons
- **Primary:** Solid Medical Cyan with Deep Navy text. No shadow, but a subtle glow on hover.
- **Ghost:** Transparent background with a 1px Medical Cyan border.

### Input Fields
Minimalist underlines or very subtle glass containers. The focus state must trigger a "scanning" animation—a horizontal cyan line that pulses once across the field.