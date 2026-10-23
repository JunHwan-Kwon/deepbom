// Presentation policy shared by MCP responses and explicit widget chat handoffs.
// A host/browser locale describes the environment, not the user's choice of
// response language. The host reads explicit requests from the conversation;
// the widget neither receives nor parses private conversation text.
export const WIDGET_LANGUAGE = "en";

export const CONVERSATION_LANGUAGE_GUIDANCE = "DEEPBOM response language: use English by default. Honor an explicit request for Korean or another response language in the current conversation, including an earlier explicit preference, until the user changes it. Do not infer a language preference from browser or host locale, location, filenames, model metadata, or the language of an automatically generated follow-up. Use the chosen language consistently for explanations and table headings. Preserve exact evidence values, code identifiers, and the English UI labels when referring to controls such as Report in chat.";

export function withConversationLanguage(message) {
  return `${CONVERSATION_LANGUAGE_GUIDANCE}\n\n${message}`;
}
