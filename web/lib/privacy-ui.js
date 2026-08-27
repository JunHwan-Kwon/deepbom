import {
  readResearchConsent,
  readAgreementAccepted,
  syncResearchConsent,
  writeAgreementAccepted,
  writeResearchConsent,
} from "./storage.js";
import { closeModal, installModalKeyboard, openModal } from "./modal-accessibility.js";

export function initPrivacyAgreementUi({
  privacyAgree,
  acceptAgreement,
  researchConsent,
  agreementBackdrop,
  body,
  onSyncError = (error) => console.warn("Consent sync skipped", error),
  onAccept = () => {},
}) {
  researchConsent.checked = readResearchConsent();
  if (readAgreementAccepted()) {
    privacyAgree.checked = true;
    acceptAgreement.disabled = false;
    closePrivacyAgreementUi({ agreementBackdrop, body });
    onAccept();
    return;
  }
  privacyAgree.checked = false;
  acceptAgreement.disabled = true;
  body.classList.add("privacy-locked");
  openModal(agreementBackdrop, { focus: privacyAgree });
  installModalKeyboard(agreementBackdrop);

  privacyAgree.addEventListener("change", () => {
    acceptAgreement.disabled = !privacyAgree.checked;
  });

  acceptAgreement.addEventListener("click", () => {
    if (!privacyAgree.checked) return;
    writeResearchConsent(researchConsent.checked);
    syncResearchConsent().catch(onSyncError);
    writeAgreementAccepted();
    closePrivacyAgreementUi({ agreementBackdrop, body });
    onAccept();
  });
}

export function closePrivacyAgreementUi({ agreementBackdrop, body }) {
  closeModal(agreementBackdrop);
  body.classList.remove("privacy-locked");
}
