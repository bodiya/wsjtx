module ft8_coherent

! Coherent FT8 decoding path.
!
! The standard decoder reduces the per-symbol tone correlations to
! magnitudes.  This module keeps them complex, estimates the complex
! channel gain over the message from the 21 Costas sync tones, scores the
! tones with a partially coherent metric that trusts the estimate only as
! far as its error variance allows, and after a failed decode reuses the
! LDPC posterior LLRs as extra pilots (decision-directed passes).
!
! The path is receive-side only and is run from ft8b after the standard
! noncoherent decode attempts fail.  It is enabled with decode depth >= 4
! or the environment variable FT8COH=1 (FT8COH=0 forces it off).
!
! Other environment knobs, for experiments:
!   FT8COH_NCAND  number of frequency candidates to try      (default 5)
!   FT8COH_NDD    decision-directed passes per candidate     (default 3)
!   FT8COH_DEBUG  =1 appends estimator diagnostics to ft8_coherent.log
!   FT8COH_MINPSN minimum sync-tone signal/noise power ratio to try (default 1.3)
!   FT8COH_MAXCHK run a DD pass only if the failed decode left at most this
!                 many unsatisfied parity checks                 (default 40)
!   FT8COH_LLGAP  try a further frequency candidate only if its search
!                 log-likelihood is within this of the best       (default 1e30)
!   FT8COH_MAXHE  accept a coherent decode only with at most this many hard
!                 errors and at most 30 unsatisfied checks        (default 30)
!   FT8COH_LLRSCALE multiplies the coherent LLRs (experiments only; default 1)

  implicit none
  private
  public :: ft8coh_state, ft8coh_enabled, ft8coh_init, ft8coh_next, ft8coh_log
  public :: ft8coh_accept

  integer, parameter :: NN=79            !Channel symbols per message
  integer, parameter :: NSYNC=21         !Costas sync symbols
  integer, parameter :: NSPS=32          !Samples per symbol at 200 Hz
  integer, parameter :: NSAMP=NN*NSPS
  integer, parameter :: NDOP=10          !Doppler-spread hypotheses
  integer, parameter :: MAXCAND=5        !Frequency candidates kept
  integer, parameter :: MAXDD=3          !Decision-directed passes
  integer, parameter :: NCOARSE=5        !Hypotheses used in the coarse search
  real, parameter :: TSYM=0.16           !Symbol period (s)
  real, parameter :: DT2=1.0/200.0       !Sample period at 200 Hz (s)
  real, parameter :: TMSG=NN*TSYM        !Message length (s)
  real, parameter :: FMAX=3.1            !Offset search half-range (Hz)
  real, parameter :: PI=3.14159265358979
  real, parameter :: DOPBANK(NDOP)=(/0.0,0.05,0.1,0.2,0.35,0.5,0.7,1.0,1.5,2.0/)
  integer, parameter :: ICOARSE(NCOARSE)=(/1,3,5,7,9/)
  integer, parameter :: ICOS7(0:6)=(/3,1,4,0,6,5,2/)
  integer, parameter :: GRAYMAP(0:7)=(/0,1,3,2,5,6,4,7/)

  type ft8coh_state
     complex :: cd(0:NSAMP-1)            !Baseband samples of the 79 symbols
     complex :: y(0:7,NN)                !Tone correlations at offset delta
     real :: sigma2                      !Noise variance per bin (normalised)
     real :: ps                          !Signal power estimate (raw units)
     real :: delta                       !Current frequency offset (Hz)
     integer :: ncand,icand,idd,idop
     real :: cand(MAXCAND),candll(MAXCAND)
     integer :: candidop(MAXCAND)
     real(8) :: r(NN,NN)                 !Channel autocorrelation, chosen hypothesis
     complex :: hhat(NN)                 !Channel estimate per symbol
     real :: sige2(NN)                   !Error variance of hhat
     real :: ptone(0:7,NN)               !Soft tone probabilities (DD passes)
     real :: f1,xdt                      !For diagnostics only
  end type ft8coh_state

  integer, save :: ncand_max=MAXCAND
  integer, save :: ndd_max=MAXDD
  integer, save :: maxchk=40
  integer, parameter :: MAXCHK_ACCEPT=30
  integer, save :: maxhe=30
  real, save :: minpsn=1.3
  real, save :: llrscale=1.0
  real, save :: llgap=1.0e30
  logical, save :: ldebug=.false.
  integer, save :: nenv=-1               !-1 not read, 0 default, 1 force on, 2 force off

contains

  logical function ft8coh_enabled(ndepth)
    integer, intent(in) :: ndepth
    character(len=16) :: val
    integer :: n,ios,ival
    real :: rval

    if(nenv.lt.0) then
       nenv=0
       call get_environment_variable('FT8COH',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0) then
             if(ival.ne.0) nenv=1
             if(ival.eq.0) nenv=2
          endif
       endif
       call get_environment_variable('FT8COH_NCAND',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0) ncand_max=max(1,min(MAXCAND,ival))
       endif
       call get_environment_variable('FT8COH_NDD',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0) ndd_max=max(0,min(MAXDD,ival))
       endif
       call get_environment_variable('FT8COH_MAXCHK',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0) maxchk=ival
       endif
       call get_environment_variable('FT8COH_MAXHE',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0) maxhe=ival
       endif
       call get_environment_variable('FT8COH_MINPSN',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) rval
          if(ios.eq.0) minpsn=rval
       endif
       call get_environment_variable('FT8COH_LLRSCALE',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) rval
          if(ios.eq.0) llrscale=rval
       endif
       call get_environment_variable('FT8COH_LLGAP',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) rval
          if(ios.eq.0) llgap=rval
       endif
       call get_environment_variable('FT8COH_DEBUG',val,n,ios)
       if(ios.eq.0 .and. n.gt.0) then
          read(val(1:n),*,iostat=ios) ival
          if(ios.eq.0 .and. ival.ne.0) ldebug=.true.
       endif
    endif
    ft8coh_enabled=.false.
    if(ndepth.ge.4) ft8coh_enabled=.true.
    if(nenv.eq.1) ft8coh_enabled=.true.
    if(nenv.eq.2) ft8coh_enabled=.false.
  end function ft8coh_enabled

!-----------------------------------------------------------------------
  subroutine ft8coh_init(cd0,np2,ibest,f1,xdt,s,ok)

! Set up the coherent state for one candidate signal: tone correlations
! at zero offset, noise and signal power, and the joint frequency-offset /
! Doppler-spread search over the sync tones.  Produces an ordered list of
! frequency candidates (best first).

    integer, intent(in) :: np2,ibest
    complex, intent(in) :: cd0(0:np2-1)
    real, intent(in) :: f1,xdt
    type(ft8coh_state), intent(inout) :: s
    logical, intent(out) :: ok

    integer :: m,i,k,n,id,ig,ng,isep,ic,j,jbest,idbest
    integer :: isync(NSYNC),itsync(NSYNC)
    integer, allocatable :: ipk(:)
    real :: ts(NSYNC),step,dfine,delta,ll,llmax,q
    real, allocatable :: llg(:),llpk(:)
    complex :: z(NSYNC),u(NSYNC)
    real(8) :: a(NSYNC,NSYNC),mm(NSYNC,NSYNC,NDOP),logdet(NDOP),ur(NSYNC),ui(NSYNC)
    logical :: lused

    s%f1=f1
    s%xdt=xdt
    s%ncand=0
    s%icand=0
    s%idd=0
    s%delta=0.0
    ok=.false.

! Samples of the 79 symbols, zero outside the available data
    do m=0,NSAMP-1
       i=ibest+m
       if(i.ge.0 .and. i.le.np2-1) then
          s%cd(m)=cd0(i)
       else
          s%cd(m)=0.0
       endif
    enddo

! Tone correlations at zero offset (raw units)
    s%ps=1.0
    call tone_corr(s)

! Noise variance per bin from the 7 off-tone bins of the 21 sync symbols
! (147 pure-noise samples when the sync is right; no order-statistic bias)
    call sync_index(isync,itsync)
    q=0.0
    do i=1,NSYNC
       do k=0,7
          if(k.eq.itsync(i)) cycle
          q=q+real(s%y(k,isync(i)))**2+aimag(s%y(k,isync(i)))**2
       enddo
    enddo
    q=q/(7*NSYNC)
    if(q.le.0.0) return
    s%sigma2=q

! Sync observations and signal power
    q=0.0
    do i=1,NSYNC
       z(i)=s%y(itsync(i),isync(i))
       q=q+real(z(i))**2+aimag(z(i))**2
    enddo
    q=q/NSYNC-s%sigma2
    if(q.lt.minpsn*s%sigma2) return
    s%ps=max(q,0.05*s%sigma2)
    s%sigma2=s%sigma2/s%ps
    s%y=s%y/sqrt(s%ps)
    z=z/sqrt(s%ps)
    do i=1,NSYNC
       ts(i)=(isync(i)-1)*TSYM
    enddo

! Per-hypothesis inverse covariance and log-determinant over the sync times
    do id=1,NDOP
       do i=1,NSYNC
          do k=1,NSYNC
             a(i,k)=gauss_corr(DOPBANK(id),ts(i)-ts(k))
          enddo
          a(i,i)=a(i,i)+s%sigma2
       enddo
       call spd_inverse(a,NSYNC,mm(:,:,id),logdet(id),ok)
       if(.not.ok) return
    enddo
    ok=.false.

! Coarse grid over the offset, profile over the coarse hypothesis subset
    step=min(0.025,0.2/TMSG)
    ng=2*nint(FMAX/step)+1
    allocate(llg(ng),ipk(ng),llpk(ng))
    do ig=1,ng
       delta=(ig-(ng+1)/2)*step
       do i=1,NSYNC
          u(i)=z(i)*cmplx(cos(2.0*PI*delta*ts(i)),-sin(2.0*PI*delta*ts(i)))
          ur(i)=real(u(i))
          ui(i)=aimag(u(i))
       enddo
       llmax=-huge(1.0)
       do j=1,NCOARSE
          id=ICOARSE(j)
          ll=-real(quad_form(mm(:,:,id),NSYNC,ur,ui)+logdet(id))
          if(ll.gt.llmax) llmax=ll
       enddo
       llg(ig)=llmax
    enddo

! Local maxima, strongest first, separated by at least 1/T
    isep=max(1,nint(1.0/TMSG/step))
    n=0
    do ig=2,ng-1
       if(llg(ig).ge.llg(ig-1) .and. llg(ig).ge.llg(ig+1)) then
          n=n+1
          ipk(n)=ig
          llpk(n)=llg(ig)
       endif
    enddo
    if(n.eq.0) then
       deallocate(llg,ipk,llpk)
       return
    endif
    call sort_desc(llpk,ipk,n)
    do j=1,n
       if(s%ncand.ge.ncand_max) exit
       lused=.false.
       do ic=1,s%ncand
          if(abs(ipk(j)-nint((s%cand(ic)/step)+(ng+1)/2)).lt.isep) lused=.true.
       enddo
       if(lused) cycle
       s%ncand=s%ncand+1
       s%cand(s%ncand)=(ipk(j)-(ng+1)/2)*step
    enddo
    deallocate(llg,ipk,llpk)

! Fine pass around each candidate over the full hypothesis bank; the
! Doppler hypothesis is chosen at the best offset.
    dfine=step/12.0
    do ic=1,s%ncand
       llmax=-huge(1.0)
       jbest=0
       idbest=1
       do j=-18,18
          delta=s%cand(ic)+j*dfine
          do i=1,NSYNC
             u(i)=z(i)*cmplx(cos(2.0*PI*delta*ts(i)),-sin(2.0*PI*delta*ts(i)))
             ur(i)=real(u(i))
             ui(i)=aimag(u(i))
          enddo
          do id=1,NDOP
             ll=-real(quad_form(mm(:,:,id),NSYNC,ur,ui)+logdet(id))
             if(ll.gt.llmax) then
                llmax=ll
                jbest=j
                idbest=id
             endif
          enddo
       enddo
       s%cand(ic)=s%cand(ic)+jbest*dfine
       s%candll(ic)=llmax
       s%candidop(ic)=idbest
    enddo
    call sort_cands(s)
    ok=s%ncand.gt.0
    if(ldebug) then
       open(78,file='ft8_coherent.log',status='unknown',position='append')
       write(78,'(a,f8.1,f6.2,a,f7.3,a,f6.2,a,i2)') 'init f1=',f1,xdt,' sig2=',s%sigma2,  &
            ' ps/n=',1.0/s%sigma2,' ncand=',s%ncand
       do ic=1,s%ncand
          write(78,'(a,i2,a,f8.4,a,f10.2,a,f5.2)') '  cand',ic,' df=',s%cand(ic),      &
               ' ll=',s%candll(ic),' dop=',DOPBANK(s%candidop(ic))
       enddo
       close(78)
    endif
    return
  end subroutine ft8coh_init

!-----------------------------------------------------------------------
  subroutine ft8coh_next(s,zpost,nchk,llr,ok)

! Produce the next set of 174 bit LLRs.  The first call for a candidate
! uses the sync-only channel estimate; subsequent calls run a
! decision-directed pass from the posterior LLRs of the failed decode
! (nchk is its number of unsatisfied parity checks).
! ok=.false. when all candidates and passes are exhausted.

    type(ft8coh_state), intent(inout) :: s
    real, intent(in) :: zpost(174)
    integer, intent(in) :: nchk
    real, intent(out) :: llr(174)
    logical, intent(out) :: ok

    integer :: isync(NSYNC),itsync(NSYNC),i
    complex :: zobs(NN)
    real :: vobs(NN)
    integer :: iobs(NN)

    ok=.false.
    llr=0.0
    if(s%icand.ge.1 .and. s%idd.lt.ndd_max .and. nchk.le.maxchk) then
       s%idd=s%idd+1
       call dd_pass(s,zpost)
    else
       s%icand=s%icand+1
       if(s%icand.gt.s%ncand) return
       if(s%candll(1)-s%candll(s%icand).gt.llgap) return
       s%idd=0
       s%delta=s%cand(s%icand)
       s%idop=s%candidop(s%icand)
       call build_r(s)
       call tone_corr(s)
       call sync_index(isync,itsync)
       do i=1,NSYNC
          iobs(i)=isync(i)
          zobs(i)=s%y(itsync(i),isync(i))
          vobs(i)=s%sigma2
       enddo
       call smooth(s,NSYNC,iobs,zobs,vobs)
    endif
    call score(s,llr)
    ok=.true.
    return
  end subroutine ft8coh_next

!-----------------------------------------------------------------------
  logical function ft8coh_accept(nharderrors,nchk)
! False-decode guard for codewords found from coherent LLRs: the OSD can
! return a valid CRC from random LLRs, and the coherent path makes many
! more attempts per candidate than the standard path.
    integer, intent(in) :: nharderrors,nchk
    ft8coh_accept=nharderrors.ge.0 .and. nharderrors.le.maxhe .and. nchk.le.MAXCHK_ACCEPT
  end function ft8coh_accept

!-----------------------------------------------------------------------
  subroutine ft8coh_log(s,msg37,nharderrors,dmin,nchk,lok)
    type(ft8coh_state), intent(in) :: s
    character*37, intent(in) :: msg37
    integer, intent(in) :: nharderrors,nchk
    real, intent(in) :: dmin
    logical, intent(in) :: lok
    character*6 tag
    if(.not.ldebug) return
    tag='TRY   '
    if(lok) tag='DECODE'
    open(78,file='ft8_coherent.log',status='unknown',position='append')
    write(78,'(a6,a,f8.1,f6.2,a,f6.2,a,i2,a,i2,a,f8.4,a,f5.2,a,i3,a,i3,f7.1,2x,a)') tag,  &
         ' f1=',s%f1,s%xdt,' ps/n=',1.0/s%sigma2,' cand=',s%icand,' dd=',s%idd,          &
         ' df=',s%delta,' dop=',DOPBANK(s%idop),' nchk=',nchk,' nhe=',nharderrors,dmin,   &
         trim(msg37)
    close(78)
  end subroutine ft8coh_log

!-----------------------------------------------------------------------
  subroutine dd_pass(s,zpost)

! One decision-directed pass: soft tone probabilities from the posterior
! bit LLRs, soft channel observations for the data symbols, residual
! offset refinement, re-smoothing with sync plus soft observations.

    type(ft8coh_state), intent(inout) :: s
    real, intent(in) :: zpost(174)

    integer :: iobs(NN),nobs,i,n,k,isym,ib,i32,j,jbest
    real :: p1(3),pb,vobs(NN),ts(NN),ll,llmax,delta,dgrid,dres
    complex :: zobs(NN),u(NN)
    real(8) :: a(NN,NN),mm(NN,NN),logdet,ur(NN),ui(NN)
    logical :: lsoft(NN),ok

! Tone probabilities for the data symbols
    s%ptone=0.0
    lsoft=.false.
    do n=8,72
       if(n.ge.37 .and. n.le.43) cycle
       if(n.le.36) then
          i32=1+(n-8)*3
       else
          i32=88+(n-44)*3
       endif
       do ib=1,3
          p1(ib)=1.0/(1.0+exp(-max(-30.0,min(30.0,zpost(i32+ib-1)))))
       enddo
       do isym=0,7
          pb=1.0
          do ib=1,3
             if(btest(isym,3-ib)) then
                pb=pb*p1(ib)
             else
                pb=pb*(1.0-p1(ib))
             endif
          enddo
          s%ptone(GRAYMAP(isym),n)=pb
       enddo
       lsoft(n)=.true.
    enddo

! Observation list: sync tones then soft data symbols
    call soft_obs(s,lsoft,nobs,iobs,zobs,vobs)

! Residual offset refinement over all observations
    do n=1,NN
       ts(n)=(n-1)*TSYM
    enddo
    do i=1,nobs
       do k=1,nobs
          a(i,k)=s%r(iobs(i),iobs(k))
       enddo
       a(i,i)=a(i,i)+vobs(i)
    enddo
    call spd_inverse(a,nobs,mm,logdet,ok)
    if(ok) then
       dres=0.0
       do j=1,2
          if(j.eq.1) then
             dgrid=0.005
             jbest=-60
             llmax=-huge(1.0)
             do i=-60,60
                delta=i*dgrid
                call quad_at(delta)
                if(ll.gt.llmax) then
                   llmax=ll
                   jbest=i
                endif
             enddo
             dres=jbest*dgrid
          else
             dgrid=0.001
             llmax=-huge(1.0)
             jbest=0
             do i=-10,10
                delta=dres+i*dgrid
                call quad_at(delta)
                if(ll.gt.llmax) then
                   llmax=ll
                   jbest=i
                endif
             enddo
             dres=dres+jbest*dgrid
          endif
       enddo
       s%delta=s%delta+dres
       call tone_corr(s)
       call soft_obs(s,lsoft,nobs,iobs,zobs,vobs)
    endif
    call smooth(s,nobs,iobs,zobs,vobs)
    return

  contains

    subroutine quad_at(d)
      real, intent(in) :: d
      integer :: ii
      do ii=1,nobs
         u(ii)=zobs(ii)*cmplx(cos(2.0*PI*d*ts(iobs(ii))),-sin(2.0*PI*d*ts(iobs(ii))))
         ur(ii)=real(u(ii))
         ui(ii)=aimag(u(ii))
      enddo
      ll=-real(quad_form(mm,nobs,ur,ui))
    end subroutine quad_at

  end subroutine dd_pass

!-----------------------------------------------------------------------
  subroutine soft_obs(s,lsoft,nobs,iobs,zobs,vobs)

! Build the observation list: the 21 sync tones (variance sigma2) followed
! by soft observations for the data symbols with usable tone probabilities.

    type(ft8coh_state), intent(in) :: s
    logical, intent(in) :: lsoft(NN)
    integer, intent(out) :: nobs,iobs(NN)
    complex, intent(out) :: zobs(NN)
    real, intent(out) :: vobs(NN)
    integer :: isync(NSYNC),itsync(NSYNC),i,n,k
    real :: q,q3
    complex :: zz

    call sync_index(isync,itsync)
    nobs=0
    do i=1,NSYNC
       nobs=nobs+1
       iobs(nobs)=isync(i)
       zobs(nobs)=s%y(itsync(i),isync(i))
       vobs(nobs)=s%sigma2
    enddo
    do n=1,NN
       if(.not.lsoft(n)) cycle
       q=0.0
       q3=0.0
       zz=0.0
       do k=0,7
          q=q+s%ptone(k,n)**2
          q3=q3+s%ptone(k,n)**3
          zz=zz+s%ptone(k,n)*s%y(k,n)
       enddo
       if(q.lt.1.0e-3) cycle
       nobs=nobs+1
       iobs(nobs)=n
       zobs(nobs)=zz/q
       vobs(nobs)=(s%sigma2*q+max(q3-q*q,0.0))/(q*q)
    enddo
    return
  end subroutine soft_obs

!-----------------------------------------------------------------------
  subroutine smooth(s,nobs,iobs,zobs,vobs)

! MMSE (Wiener) smoothing of the channel gain over all 79 symbols from
! the observations: hhat = R (R+V)^-1 z, sige2 = 1 - diag(R (R+V)^-1 R).

    type(ft8coh_state), intent(inout) :: s
    integer, intent(in) :: nobs,iobs(NN)
    complex, intent(in) :: zobs(NN)
    real, intent(in) :: vobs(NN)
    real(8) :: a(NN,NN),wr(NN),wi(NN),g(NN,NN),rr(NN,NN)
    real(8) :: hr,hi,e
    integer :: i,k,n
    logical :: ok

    do i=1,nobs
       do k=1,nobs
          a(i,k)=s%r(iobs(i),iobs(k))
       enddo
       a(i,i)=a(i,i)+vobs(i)
       wr(i)=real(zobs(i))
       wi(i)=aimag(zobs(i))
       do n=1,NN
          rr(i,n)=s%r(iobs(i),n)
       enddo
    enddo
    call cholesky(a,nobs,ok)
    if(.not.ok) then
       s%hhat=0.0
       s%sige2=1.0
       return
    endif
    call chol_solve(a,nobs,wr)
    call chol_solve(a,nobs,wi)
    do n=1,NN
       g(1:nobs,n)=rr(1:nobs,n)
       call chol_solve(a,nobs,g(1:nobs,n))
    enddo
    do n=1,NN
       hr=0.d0
       hi=0.d0
       e=0.d0
       do i=1,nobs
          hr=hr+rr(i,n)*wr(i)
          hi=hi+rr(i,n)*wi(i)
          e=e+rr(i,n)*g(i,n)
       enddo
       s%hhat(n)=cmplx(real(hr),real(hi))
       s%sige2(n)=real(max(1.d0-e,0.d0))
    enddo
    return
  end subroutine smooth

!-----------------------------------------------------------------------
  subroutine score(s,llr)

! Partially coherent tone metric and Gray-mapped bit LLRs (positive
! LLR means bit=1, the convention of the WSJT-X LDPC decoders).

    type(ft8coh_state), intent(in) :: s
    real, intent(out) :: llr(174)
    integer :: n,k,isym,ib,i32
    real :: m(0:7),ms(0:7),v,g1,g2,x1(4),x0(4)
    integer :: n1,n0

    llr=0.0
    do n=8,72
       if(n.ge.37 .and. n.le.43) cycle
       if(n.le.36) then
          i32=1+(n-8)*3
       else
          i32=88+(n-44)*3
       endif
       v=s%sigma2+s%sige2(n)
       g1=2.0/v
       g2=1.0/s%sigma2-1.0/v
       do k=0,7
          m(k)=g1*(real(s%hhat(n))*real(s%y(k,n))+aimag(s%hhat(n))*aimag(s%y(k,n)))  &
               +g2*(real(s%y(k,n))**2+aimag(s%y(k,n))**2)
       enddo
       do isym=0,7
          ms(isym)=m(GRAYMAP(isym))
       enddo
       do ib=1,3
          n1=0
          n0=0
          do isym=0,7
             if(btest(isym,3-ib)) then
                n1=n1+1
                x1(n1)=ms(isym)
             else
                n0=n0+1
                x0(n0)=ms(isym)
             endif
          enddo
          llr(i32+ib-1)=llrscale*(lse4(x1)-lse4(x0))
       enddo
    enddo
    return
  end subroutine score

  real function lse4(x)
    real, intent(in) :: x(4)
    real :: xm
    xm=maxval(x)
    lse4=xm+log(exp(x(1)-xm)+exp(x(2)-xm)+exp(x(3)-xm)+exp(x(4)-xm))
  end function lse4

!-----------------------------------------------------------------------
  subroutine tone_corr(s)

! Tone correlations Y_k(n), k=0..7, at the current offset s%delta: derotate
! the samples by exp(-j 2 pi delta t) and take the 8 lowest bins of the
! 32-point DFT of each symbol.  Normalised by sqrt(ps).

    type(ft8coh_state), intent(inout) :: s
    complex :: tw(0:7,0:NSPS-1),x(0:NSPS-1),rot,drot,acc
    integer :: n,k,m
    real :: dphi,fac

    do k=0,7
       do m=0,NSPS-1
          tw(k,m)=cmplx(cos(2.0*PI*k*m/NSPS),-sin(2.0*PI*k*m/NSPS))
       enddo
    enddo
    dphi=-2.0*PI*s%delta*DT2
    drot=cmplx(cos(dphi),sin(dphi))
    rot=cmplx(1.0,0.0)
    fac=1.0/sqrt(s%ps)
    do n=1,NN
       do m=0,NSPS-1
          x(m)=s%cd((n-1)*NSPS+m)*rot
          rot=rot*drot
       enddo
       do k=0,7
          acc=0.0
          do m=0,NSPS-1
             acc=acc+x(m)*tw(k,m)
          enddo
          s%y(k,n)=acc*fac
       enddo
    enddo
    return
  end subroutine tone_corr

!-----------------------------------------------------------------------
  subroutine build_r(s)
    type(ft8coh_state), intent(inout) :: s
    integer :: i,k
    do i=1,NN
       do k=1,NN
          s%r(i,k)=gauss_corr(DOPBANK(s%idop),(i-k)*TSYM)
       enddo
    enddo
  end subroutine build_r

  real(8) function gauss_corr(spread,tau)
! Channel autocorrelation for a Gaussian Doppler spectrum (Watterson),
! sigma_d = spread/2.
    real, intent(in) :: spread,tau
    real(8) :: sd
    sd=0.5d0*spread
    gauss_corr=exp(-2.d0*PI*PI*sd*sd*dble(tau)*dble(tau))
  end function gauss_corr

  subroutine sync_index(isync,itsync)
! 1-based symbol indices of the 21 Costas sync symbols and their tones
    integer, intent(out) :: isync(NSYNC),itsync(NSYNC)
    integer :: j
    do j=1,7
       isync(j)=j
       isync(j+7)=36+j
       isync(j+14)=72+j
       itsync(j)=ICOS7(j-1)
       itsync(j+7)=ICOS7(j-1)
       itsync(j+14)=ICOS7(j-1)
    enddo
  end subroutine sync_index

  real(8) function quad_form(mm,n,ur,ui)
! Re(u^H M u) for real symmetric M
    integer, intent(in) :: n
    real(8), intent(in) :: mm(:,:),ur(:),ui(:)
    integer :: i,k
    real(8) :: sr,si
    quad_form=0.d0
    do i=1,n
       sr=0.d0
       si=0.d0
       do k=1,n
          sr=sr+mm(k,i)*ur(k)
          si=si+mm(k,i)*ui(k)
       enddo
       quad_form=quad_form+ur(i)*sr+ui(i)*si
    enddo
  end function quad_form

!-----------------------------------------------------------------------
! Dense linear algebra for small symmetric positive definite systems

  subroutine cholesky(a,n,ok)
! In-place lower Cholesky factor of the leading n x n block of a
    real(8), intent(inout) :: a(:,:)
    integer, intent(in) :: n
    logical, intent(out) :: ok
    integer :: i,j,k
    real(8) :: sum
    ok=.false.
    do j=1,n
       sum=a(j,j)
       do k=1,j-1
          sum=sum-a(j,k)*a(j,k)
       enddo
       if(sum.le.0.d0) return
       a(j,j)=sqrt(sum)
       do i=j+1,n
          sum=a(i,j)
          do k=1,j-1
             sum=sum-a(i,k)*a(j,k)
          enddo
          a(i,j)=sum/a(j,j)
       enddo
    enddo
    ok=.true.
  end subroutine cholesky

  subroutine chol_solve(a,n,b)
! Solve (L L^T) x = b in place, L the lower factor from cholesky
    real(8), intent(in) :: a(:,:)
    integer, intent(in) :: n
    real(8), intent(inout) :: b(:)
    integer :: i,k
    real(8) :: sum
    do i=1,n
       sum=b(i)
       do k=1,i-1
          sum=sum-a(i,k)*b(k)
       enddo
       b(i)=sum/a(i,i)
    enddo
    do i=n,1,-1
       sum=b(i)
       do k=i+1,n
          sum=sum-a(k,i)*b(k)
       enddo
       b(i)=sum/a(i,i)
    enddo
  end subroutine chol_solve

  subroutine spd_inverse(a,n,ainv,logdet,ok)
! Inverse and log-determinant of a symmetric positive definite matrix
    real(8), intent(inout) :: a(:,:)
    integer, intent(in) :: n
    real(8), intent(out) :: ainv(:,:),logdet
    logical, intent(out) :: ok
    integer :: i,j
    real(8) :: col(n)
    call cholesky(a,n,ok)
    logdet=0.d0
    if(.not.ok) return
    do i=1,n
       logdet=logdet+2.d0*log(a(i,i))
    enddo
    do j=1,n
       col=0.d0
       col(j)=1.d0
       call chol_solve(a,n,col)
       ainv(1:n,j)=col
    enddo
  end subroutine spd_inverse

  subroutine sort_desc(v,idx,n)
! Sort v descending, carrying idx along (insertion sort, n small)
    integer, intent(in) :: n
    real, intent(inout) :: v(n)
    integer, intent(inout) :: idx(n)
    integer :: i,j,it
    real :: t
    do i=2,n
       t=v(i)
       it=idx(i)
       j=i-1
       do while(j.ge.1)
          if(v(j).ge.t) exit
          v(j+1)=v(j)
          idx(j+1)=idx(j)
          j=j-1
       enddo
       v(j+1)=t
       idx(j+1)=it
    enddo
  end subroutine sort_desc

  subroutine sort_cands(s)
    type(ft8coh_state), intent(inout) :: s
    integer :: i,j,it
    real :: t,tc
    do i=2,s%ncand
       t=s%candll(i)
       tc=s%cand(i)
       it=s%candidop(i)
       j=i-1
       do while(j.ge.1)
          if(s%candll(j).ge.t) exit
          s%candll(j+1)=s%candll(j)
          s%cand(j+1)=s%cand(j)
          s%candidop(j+1)=s%candidop(j)
          j=j-1
       enddo
       s%candll(j+1)=t
       s%cand(j+1)=tc
       s%candidop(j+1)=it
    enddo
  end subroutine sort_cands

end module ft8_coherent
