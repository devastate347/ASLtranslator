import Head from 'next/head';
import styles from '/styles/Home.module.css';

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Bringing ASL to Everyone</title>
      </Head>
      
      <header className={styles.header}>
        <h1>Brining ASL to the World</h1>
      </header>

      <section className={styles.body}>
        <h2>Body Title</h2>
        <p></p>
      </section>

      <footer className={styles.footer}>
        <p>&copy; {new Date().getFullYear()} Your Name</p>
      </footer>
    </div>
  );
}
