import './App.css'
import Home from './Pages/Home'
import Navbar from './components/Navbar'

function App() {
  return (
    <div className="min-h-screen w-full">
      <main className="bg-[hsl(24,12%,8%)] min-h-screen w-full text-white">
        <Navbar />
        <Home />
      </main>
    </div>
  );
}

export default App