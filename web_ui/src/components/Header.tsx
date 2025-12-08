type HeaderProps = {
  apiHealthy: boolean | null
  isCameraOn: boolean
}

export function Header({ apiHealthy, isCameraOn }: HeaderProps) {
  return (
    <header className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">
          Kedip • Blink Detection
        </p>
        <h1 className="text-3xl font-semibold leading-tight md:text-4xl">Inference in your browser.</h1>
        <p className="max-w-2xl text-sm text-neutral-500 md:text-base">
          Use Kedip and see how long you can hold your stare.
        </p>
      </div>

      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 rounded-full border border-neutral-200 px-3 py-1.5 shadow-sm">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${
              apiHealthy === null ? 'bg-neutral-300' : apiHealthy ? 'bg-emerald-500' : 'bg-red-500'
            }`}
          />
          <span className="text-xs font-medium text-neutral-700">
            {apiHealthy === false ? 'API offline' : apiHealthy === true ? 'API connected' : 'API check'}
          </span>
        </div>
        <div className="flex items-center gap-2 rounded-full border border-neutral-200 px-3 py-1.5 shadow-sm">
          <span className={`inline-block h-2.5 w-2.5 rounded-full ${isCameraOn ? 'bg-emerald-500' : 'bg-red-500'}`} />
          <span className="text-xs font-medium text-neutral-700">
            {isCameraOn ? 'Camera on' : 'Camera idle'}
          </span>
        </div>
      </div>
    </header>
  )
}
