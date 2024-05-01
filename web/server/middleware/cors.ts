export default defineEventHandler((event) => {
  const headers = {
    'Access-Control-Allow-Origin': '*',
  }
  setHeaders(event, headers)
})
