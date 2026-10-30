import AdminContent from './AdminContent';

export const metadata = {
  title: 'Administration',
};

/** Administration at /kb/admin (Users, Logs & monitoring, System). */
export default function AdminPage() {
  return <AdminContent appName={process.env.APP_NAME || 'Growlyze Multimodal AI Chat'} />;
}
