import IntegrationsContent from './IntegrationsContent';

export const metadata = {
  title: 'Integrations',
};

/**
 * Integrations at /kb/integrations. Also where the Google Drive sign-in
 * returns: GET /kb/api/drive/callback redirects here with ?drive=connected
 * (or ?drive=error&message=…).
 */
export default function IntegrationsPage() {
  return <IntegrationsContent />;
}
