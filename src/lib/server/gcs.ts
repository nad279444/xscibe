import 'server-only'
import { Storage } from '@google-cloud/storage'

let storage: Storage | undefined

export function audioBucket() {
  const bucketName = process.env.GCS_BUCKET_NAME
  const clientEmail = process.env.CLIENT_EMAIL
  const privateKey = process.env.PRIVATE_KEY?.replace(/\\n/g, '\n')
  if (!bucketName || !clientEmail || !privateKey) {
    throw new Error('GCS_BUCKET_NAME, CLIENT_EMAIL, and PRIVATE_KEY are required')
  }
  storage ??= new Storage({
    projectId: process.env.GCS_PROJECT_ID,
    credentials: { client_email: clientEmail, private_key: privateKey },
  })
  return storage.bucket(bucketName)
}
